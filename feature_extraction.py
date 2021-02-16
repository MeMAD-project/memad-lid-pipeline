#!/usr/bin/env python3
# coding=utf-8
"""Utility functions for feature extraction."""

import lidbox.data.steps as ds_steps
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf
from lidbox.features import audio
from sklearn.preprocessing import normalize

np_rng = np.random.default_rng(1)
tf.random.set_seed(np_rng.integers(0, tf.int64.max))
TF_AUTOTUNE = tf.data.experimental.AUTOTUNE


def metadata_to_dataset_input(df):
    return {
        "id": tf.constant(df.index, tf.string),
        "path": tf.constant(df.path, tf.string),
        "label": tf.constant(df.label, tf.string),
        "target": tf.constant(df.target, tf.int32),
        "split": tf.constant(df.split, tf.string),
    }


def read_audio(x):
    s, r = audio.read_wav(x["path"])
    out_rate = 16000
    s, out_rate = audio.pyfunc_resample(s, r, out_rate)
    s = audio.peak_normalize(s, dBFS=-3.0)
    s = audio.remove_silence(s, out_rate)
    return dict(x, signal=s, sample_rate=out_rate)


def random_filter(x):
    def scipy_filter(s, N=10):
        b = np_rng.normal(0, 1, N)
        return scipy.signal.lfilter(b, 1.0, s).astype(np.float32), b

    s, _ = tf.numpy_function(
        scipy_filter, [x["signal"]], [tf.float32, tf.float64], name="np_random_filter"
    )
    s = tf.cast(s, tf.float32)
    s = audio.peak_normalize(s, dBFS=-3.0)
    return dict(x, signal=s)


def create_chunks(ds, length=2000, step=1500):
    ds = ds_steps.repeat_too_short_signals(ds, length)
    return ds_steps.create_signal_chunks(ds, length, step)


def sum_and_normalize(pred):
    v = np.stack(pred).sum(axis=0)
    v = normalize(v.reshape((1, -1)), axis=1)
    return np.squeeze(v)


def embeddings_to_dataframe(ids, embeddings):
    return (pd.DataFrame.from_dict({"id": ids, "embedding": embeddings})
            .set_index("id", drop=True, verify_integrity=True)
            .sort_index())

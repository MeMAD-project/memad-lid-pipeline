#!/usr/bin/env python3
# coding=utf-8
"""Utility functions for dealing with back-end classifier models trained with sklearn."""
import collections
import os

import joblib
import numpy as np

np_rng = np.random.default_rng(1)


def embeddings_as_numpy_data(df):
    """Transform embedding vectors in Pandas dataframe into numpy arrays for sklearn."""
    X = np.stack(df.embedding.values).astype(np.float32)
    y = df.target.to_numpy(dtype=np.int32)
    return X, y


def random_sample(X, y, sample_size_ratio):
    """Random sampling from numpy arrays with given ratio. Used mainly in visualizations."""
    N = X.shape[0]
    sample_size = int(sample_size_ratio * N)
    sample_idx = np_rng.choice(np.arange(N), size=sample_size, replace=False)
    return X[sample_idx], y[sample_idx]


def pipeline_to_disk(joblib_dir, sklearn_objects):
    """Save trained sklearn model to disk."""
    os.makedirs(joblib_dir, exist_ok=True)
    for key, obj in sklearn_objects.items():
        joblib_fname = os.path.join(joblib_dir, key + ".joblib")
        print(f"Writing scikit-learn object '{obj}' to '{joblib_fname}'")
        joblib.dump(obj, joblib_fname)
    return joblib_dir


def pipeline_from_disk(joblib_dir):
    """Load trained sklearn model from disk."""
    if not os.path.isdir(joblib_dir):
        print(f"Directory '{joblib_dir}' does not exist, cannot load pipeline from disk")
        return {}
    sklearn_objects = {}
    for f in os.scandir(joblib_dir):
        if not f.name.endswith(".joblib"):
            continue
        print(f"Loading scikit-learn object from file '{f.path}'")
        key = f.name.split(".joblib")[0]
        sklearn_objects[key] = joblib.load(f.path)
    return sklearn_objects


def generate_lang2target(meta, langs):
    """
    Generate a unique language-to-target mapping,
    where integer targets are the enumeration of language labels in lexicographic order.
    """
    label2target = collections.OrderedDict(
            (l, t) for t, l in enumerate(sorted(langs)))
    meta["target"] = np.array([label2target[l] for l in meta.label], np.int32)
    return meta, label2target


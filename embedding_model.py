#!/usr/bin/env python3
# coding=utf-8
"""Embedding model definition in Tensorflow."""
import numpy as np
from tensorflow.python.framework.tensor_conversion_registry import get

np_rng = np.random.default_rng(1)
import tensorflow as tf

tf.random.set_seed(np_rng.integers(0, tf.int64.max))

from lidbox.util import model2function
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Layer,
)
from tensorflow.keras.models import Model, load_model

TIME_AXIS = 1
STDDEV_SQRT_MIN_CLIP = 1e-10


class GlobalMeanStddevPooling1D(Layer):
    """
    Compute arithmetic mean and standard deviation of the inputs along the time steps dimension,
    then output the concatenation of the computed stats.
    """

    def call(self, inputs):
        means = tf.math.reduce_mean(inputs, axis=TIME_AXIS, keepdims=True)
        variances = tf.math.reduce_mean(tf.math.square(inputs - means), axis=TIME_AXIS)
        means = tf.squeeze(means, TIME_AXIS)
        stddevs = tf.math.sqrt(
            tf.clip_by_value(variances, STDDEV_SQRT_MIN_CLIP, variances.dtype.max)
        )
        return tf.concat((means, stddevs), axis=TIME_AXIS)


def FrameLayer(
    inputs, filters, kernel_size, stride, name="frame", activation="relu", dropout_rate=None
):
    """Batch normalized temporal convolution"""
    x = Conv1D(
        filters, kernel_size, stride, name="{}_conv".format(name), activation=None, padding="same"
    )(inputs)
    x = BatchNormalization(name="{}_bn".format(name))(x)
    x = Activation(activation, name="{}_{}".format(name, str(activation)))(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate, name="{}_dropout".format(name))(x)
    return x


def SegmentLayer(inputs, units, name="segment", activation="relu", dropout_rate=None):
    """Batch normalized dense layer"""
    x = Dense(units, name="{}_dense".format(name), activation=None)(inputs)
    x = BatchNormalization(name="{}_bn".format(name))(x)
    x = Activation(activation, name="{}_{}".format(name, str(activation)))(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate, name="{}_dropout".format(name))(x)
    return x


def create(input_shape, num_outputs, output_activation="log_softmax", dropout_rate=None):
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    x = FrameLayer(x, 512, 5, 1, name="frame1", dropout_rate=dropout_rate)
    x = FrameLayer(x, 512, 3, 2, name="frame2", dropout_rate=dropout_rate)
    x = FrameLayer(x, 512, 3, 3, name="frame3", dropout_rate=dropout_rate)
    x = FrameLayer(x, 512, 1, 1, name="frame4")
    x = FrameLayer(x, 1500, 1, 1, name="frame5")

    x = GlobalMeanStddevPooling1D(name="stats_pooling")(x)

    x = SegmentLayer(x, 512, name="segment1")
    x = SegmentLayer(x, 512, name="segment2")
    x = Dense(num_outputs, name="output", activation=None)(x)

    outputs = x
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(
            outputs
        )
    return Model(inputs=inputs, outputs=outputs, name="x-vector-javascript")


def predict(model, inputs):
    return model.predict(inputs)


def get_embedding_extractor(model_path):
    model = load_model(
        model_path,
        custom_objects={
            "GlobalMeanStddevPooling1D": GlobalMeanStddevPooling1D,
            "log_softmax_v2": tf.nn.log_softmax,
        },
    )
    xvec_layer = model.get_layer(name="segment1_dense")
    xvec_layer.activation = None
    return model2function(tf.keras.Model(inputs=model.inputs, outputs=xvec_layer.output))

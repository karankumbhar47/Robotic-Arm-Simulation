"""Regression module."""

import tensorflow as tf


class Regression(tf.keras.Model):
  """Regression module."""

  def __init__(self):
    """Initialize a 3-layer MLP."""
    super(Regression, self).__init__()
    self.fc1 = tf.keras.layers.Dense(
        units=32,
        input_shape=(None, 1),
        kernel_initializer="normal",
        bias_initializer="normal",
        activation="relu")
    self.fc2 = tf.keras.layers.Dense(
        units=32,
        kernel_initializer="normal",
        bias_initializer="normal",
        activation="relu")
    self.fc3 = tf.keras.layers.Dense(
        units=1,
        kernel_initializer="normal",
        bias_initializer="normal")

  def __call__(self, x):
    return self.fc3(self.fc2(self.fc1(x)))

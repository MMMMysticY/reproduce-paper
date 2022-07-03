# encoding=utf-8
import tensorflow as tf
import modeling


class FeedForward(tf.keras.Model):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.input_dim = int(dim * mult)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(units=self.input_dim, use_bias=False)
        self.dense2 = tf.keras.layers.Dense(units=self.dim, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        # [..., dim]
        x = self.layer_norm(inputs)
        x = self.dense1(x)
        # [..., input_dim]
        x = modeling.gelu(x)
        y = self.dense2(x)
        # [..., dim]
        return y

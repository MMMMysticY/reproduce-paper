# encoding=utf-8
import tensorflow as tf
import modeling


class FeedForward(tf.keras.Model):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.input_dim = int(dim * mult)
        self.dense1 = tf.keras.layers.Dense(units=self.input_dim, kernel_initializer=tf.keras.initializers.RandomNormal())
        self.dense2 = tf.keras.layers.Dense(units=self.dim, kernel_initializer=tf.keras.initializers.RandomNormal())
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        # [..., dim]
        x = inputs
        x = self.dense1(x)
        # [..., input_dim]

        x = modeling.gelu(x)

        y = self.dense2(x)
        # [..., dim]
        y = self.layer_norm(y)
        return y

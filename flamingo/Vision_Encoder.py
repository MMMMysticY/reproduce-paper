import tensorflow as tf
import modeling


class VisionEncoder(tf.keras.Model):
    def __init__(self, dim, seq_len_image):
        super(VisionEncoder, self).__init__()
        self.dim = dim
        self.seq_len_image = seq_len_image
        self.inner_dim = self.dim * self.seq_len_image
        self.resnet = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False)
        self.resnet.trainable = False
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_proj = tf.keras.layers.Dense(
            units=self.inner_dim,
            use_bias=False,
            name='image_output_layer',
            kernel_initializer=modeling.create_initializer()
        )

    def call(self, inputs, training=None, mask=None):
        """
        输入：
            image: [batch, height, width, channel] 输入图像的[0, 255]表示
        输出:
            image_embedding: [batch, seq_len_image, dim]
        """
        image = inputs
        image_shape = modeling.get_shape_list(image, expected_rank=4)
        batch = image_shape[0]
        x = self.resnet(image)
        x = self.pool(x)
        # 【batch, some_dim]
        x = self.output_proj(x)
        # [batch, inner_dim]
        output = tf.reshape(x, shape=(batch, self.seq_len_image, self.dim))
        # [batch, seq_len_image, dim]
        return output


def main():
    batch = 32
    height = 800
    width = 800
    channel = 3
    dim = 768
    seq_len_image = 256
    input_ = tf.random.normal(shape=(batch, height, width, channel))
    vision_encoder = VisionEncoder(dim=dim, seq_len_image=seq_len_image)
    output = vision_encoder(input_)
    print(output)


if __name__ == '__main__':
    main()

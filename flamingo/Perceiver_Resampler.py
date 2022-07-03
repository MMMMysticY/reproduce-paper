# encoding=utf-8
import tensorflow as tf
import modeling
import utils


# 目前仅实现单图像位置+多张图像的效果 即论文中的[N=1, T>=1, H, W, C]
# 以下实现方法中 维度N不存在 故输入图像的维度是[T>=1, H, W, C]

class PerceiverAttention(tf.keras.Model):
    """perceiver attention方法 对image_feature和latent_queries进行cross attention"""
    def __init__(self, dim, n_head, head_dim, initializer_range=0.2):
        """
        参数:
            dim: image_feature和latent_queries的特征维度
            n_head: multi-head-attention的head个数
            head_dim: multi-head-attention每个head的维度
            initializer_range: 参数初始化方法的参数
        """
        super(PerceiverAttention, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.initializer_range = initializer_range
        self.inner_dim = self.n_head * self.head_dim  # inner_dim是n_head * head_dim之后将要被reshape为n_head和head_dim
        self.scale = self.head_dim ** -0.5  # scale是1/根号下dk
        self.image_feature_layer_norm = tf.keras.layers.LayerNormalization()  # 对image_feature进行layer norm
        self.latent_queries_layer_norm = tf.keras.layers.LayerNormalization()  # 对latent queries进行layer norm
        self.query_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            use_bias=False,
            name='query',
            kernel_initializer=modeling.create_initializer(initializer_range)
        )
        self.key_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            use_bias=False,
            name='key',
            kernel_initializer=modeling.create_initializer(initializer_range)
        )
        self.value_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            use_bias=False,
            name='value',
            kernel_initializer=modeling.create_initializer(initializer_range)
        )
        self.output_layer = tf.keras.layers.Dense(
            units=self.dim,
            use_bias=False,
            name='output',
            kernel_initializer=modeling.create_initializer(initializer_range)
        )

    def transpose_for_scores(self, input_tensor, batch_size, n_head, seq_length, head_dim):
        """
        该方法对对input_tensor进行reshape
        input_tensor [batch, seq_len, dim] -> [batch, seq_len, n_head, head_dim] -> [batch, n_head, seq_len, head_dim]
        """
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, n_head, head_dim])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def call(self, inputs, training=None, mask=None):
        """
        输入:
            image_feature: 原始图像经过vision encoder并flatten后的特征 [batch, seq_len_img, dim]
            latent_queries: 潜在query 用于对图像特征做cross-attention 进行降维 [batch, seq_len_target, dim]
        输出:
            output: perceiver_attention计算后的结果 维度和latent_queries相同 [batch, seq_len_target, dim]
        """
        image_feature = inputs[0]
        latent_queries = inputs[1]
        image_feature = self.image_feature_layer_norm(image_feature)
        latent_queries = self.latent_queries_layer_norm(latent_queries)
        # 进行layer norm

        image_feature_shape = modeling.get_shape_list(image_feature, expected_rank=3)
        latent_queries_shape = modeling.get_shape_list(latent_queries, expected_rank=3)
        assert image_feature_shape[0] == latent_queries_shape[0]
        assert image_feature_shape[-1] == latent_queries_shape[-1] == self.dim
        batch = image_feature_shape[0]
        seq_len_img = image_feature_shape[1]
        seq_len_target = latent_queries_shape[1]
        # 计算维度

        query = self.query_layer(latent_queries)
        # latent_queries作为query [batch, seq_len_target, inner_dim]
        kv_input = tf.concat([image_feature, latent_queries], axis=1)
        # image_feature和latent_queries进行拼接作为key value的输入 [batch, seq_len_target+seq_len_img, dim]
        key = self.key_layer(kv_input)
        value = self.value_layer(kv_input)
        # key和value是[batch, seq_len_target+seq_len_img, inner_dim]

        query = self.transpose_for_scores(query, batch, self.n_head, seq_len_target, self.head_dim)
        key = self.transpose_for_scores(key, batch, self.n_head, seq_len_target + seq_len_img, self.head_dim)
        value = self.transpose_for_scores(value, batch, self.n_head, seq_len_target + seq_len_img, self.head_dim)
        # query [batch, n_head, seq_len_target, head_dim]
        # key value [batch, n_head, seq_len_target+seq_len_img, head_dim]

        query = query * self.scale
        # query = query / 根号下dk

        attention_score = tf.einsum('...ik,...jk->...ij', query, key)
        # i->seq_len_target j->seq_len_target+seq_len_img k->head_dim
        # attention_score [batch, n_head, seq_len_target, seq_len_target+seq_len_img]

        # 本方法中不存在attention mask

        attention_probs = tf.nn.softmax(attention_score)
        # attention_score经过softmax得到概率分布
        # attention_probs [batch, n_head, seq_len_target, seq_len_target+seq_len_img]

        output = tf.einsum('...ij,...jk->...ik', attention_probs, value)
        # i->seq_len_target j->seq_len_target+seq_len_img k->head_dim]
        # output [batch, n_head, seq_len_target, head_dim]

        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, shape=(batch, seq_len_target, self.inner_dim))
        output = self.output_layer(output)
        # output [batch, seq_len_target, dim]
        return output


class PerceiverResampler(tf.keras.Model):
    """实现PerceiverResampler 进行图像信息(image feature)和latent_queries的多层cross attention + feedforward"""
    def __init__(self, dim, num_layers, num_latent=64, n_head=8, head_dim=64, ffw_mult=4):
        """
        Args:
            dim: 模型中的特征维度 latent_queries和image_feature的维度都是dim
            num_layers: PerceiverAttention + FeedForward的叠加层数
            n_head: cross attention的head个数
            head_dim: cross attention的每个head的维度
            ffw_mult: feedforward的隐层神经元是dim的ffw_mult倍
        """
        super(PerceiverResampler, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_latent = num_latent
        self.n_head = n_head
        self.head_dim = head_dim
        self.ffw_mult = ffw_mult
        self.latents = tf.Variable(tf.random.normal(shape=(1, self.num_latent, self.dim)))
        # 每个单独的latent的维度是 [1, seq_len_target, dim]

        # TODO 这里使用python原生list 效率可能会变低 tf应该有更合适的写法
        self.perceiver_attention_layers = list()
        self.feedforward_layers = list()
        for _ in range(self.num_layers):
            self.perceiver_attention_layers.append(
                PerceiverAttention(dim=self.dim, n_head=self.n_head, head_dim=self.head_dim))
            self.feedforward_layers.append(utils.FeedForward(dim=self.dim, mult=self.ffw_mult))
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        """
        输入:
            image_feature 原始图像经过vision encoder并flatten后的特征 [batch, seq_len_img, dim]
        输出:
            latent_queries 图像特征的整体融合特征向量 [batch, seq_len_target, dim]
        """
        image_feature_shape = inputs.shape
        batch = image_feature_shape[0]
        assert self.dim == image_feature_shape[2]

        image_feature = inputs
        # image_feature [batch, seq_len_image, dim]
        latent_queries = tf.repeat(self.latents, repeats=batch, axis=0)
        # latent_queries [batch, seq_len_target, dim]

        for num in range(self.num_layers):
            attn = self.perceiver_attention_layers[num]
            ffw = self.feedforward_layers[num]
            latent_queries = attn((image_feature, latent_queries)) + latent_queries
            latent_queries = ffw(latent_queries) + latent_queries
        latent_queries = self.layer_norm(latent_queries)
        return latent_queries


def main():
    batch = 32
    seq_len_image = 150
    seq_len_target = 64
    dim = 128
    n_head = 8
    head_dim = 32
    ffw_mult = 4
    num_layers = 2
    perceiver_resampler = PerceiverResampler(dim, num_layers, num_latent=seq_len_target, n_head=n_head,
                                             head_dim=head_dim, ffw_mult=ffw_mult)
    image_feature = tf.random.normal(shape=(batch, seq_len_image, dim))
    output = perceiver_resampler(image_feature)
    print(output)
    print(perceiver_resampler.trainable_weights)


if __name__ == '__main__':
    main()

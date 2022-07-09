# encoding=utf-8
import tensorflow as tf
import modeling
import utils


# 目前未实现论文中的media mask部分即文本仅对相对应的前一个图像做cross attention 因为本任务中仅有一个图片位置(该位置可以有多张图像)
# 则本质上的CrossAttentionWithMediaMask和Perceiver_Resampler的PerceiverAttention没有本质区别
# 后续任务若需要media mask 即需要多个图片位置和对应的文本，则需要加入 attend_media_mask控制是否使用media_mask


class CrossAttentionWithMediaMask(tf.keras.Model):
    """keras风格带有media mask的cross attention方法"""
    def __init__(self, dim, n_head, head_dim, attend_media_mask=False, initializer_range=0.2):
        """
        参数:
            dim: cross attention变量的feature维度
            n_head: multi-head attention的head个数
            head_dim: 每个head的维度
            attend_media_mask: 是否对media进行mask
            initializer_range: 全连接层参数初始化参数
        """
        super(CrossAttentionWithMediaMask, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.attend_media_mask = False
        self.initializer_range = initializer_range
        self.inner_dim = self.n_head * self.head_dim  # inner_dim是n_head * head_dim之后将要被reshape为n_head和head_dim
        self.scale = self.head_dim ** -0.5  # scale是1/根号下dk
        self.query_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            use_bias=False,
            name='query',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
        )
        self.key_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            use_bias=False,
            name='key',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
        )
        self.value_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            use_bias=False,
            name='value',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
        )
        self.output_layer = tf.keras.layers.Dense(
            units=self.dim,
            use_bias=False,
            name='output',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()

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
            visual_feature: 所有图像经过Perceiver_Resampler后得到的特征向量 [batch, seq_len_vision, dim]
            language_feature: 输入的文本信息特征 [batch, seq_len_language, dim]
            media_location: 图像对应的位置 [batch, seq_len_language] 默认为None
        输出:
            output: cross-attention的计算结果[batch, seq_len_language, dim]
        """
        visual_feature = inputs[0]
        language_feature = inputs[1]
        media_location = inputs[2]

        visual_feature_shape = modeling.get_shape_list(visual_feature, expected_rank=3)
        language_feature_shape = modeling.get_shape_list(language_feature, expected_rank=3)
        # assert visual_feature_shape[0] == language_feature_shape[0]
        # assert visual_feature_shape[2] == language_feature_shape[2]
        batch = visual_feature_shape[0]
        seq_len_vision = visual_feature_shape[1]
        seq_len_language = language_feature_shape[1]

        language_feature = self.layer_norm(language_feature)
        # 仅有一个layer norm的原因是visual_feature在Perceiver_Resampler最后一层已经layer norm过了
        # 当前仅对language_feature进行layer norm

        query = self.query_layer(language_feature)
        # language_feature作为query [batch, seq_len_language, dim]
        key = self.key_layer(visual_feature)
        value = self.value_layer(visual_feature)
        # visual_feature作为key和value [batch, seq_len_vision, dim]

        query = self.transpose_for_scores(query, batch, self.n_head, seq_len_language, self.head_dim)
        key = self.transpose_for_scores(key, batch, self.n_head, seq_len_vision, self.head_dim)
        value = self.transpose_for_scores(value, batch, self.n_head, seq_len_vision, self.head_dim)
        # query [batch, n_head, seq_len_language, head_dim]
        # key value [batch, n_head, seq_len_vision, head_dim]

        query = query * self.scale
        # query = query / 根号下dk

        attention_score = tf.einsum('...ik,...jk->...ij', query, key)
        # i->seq_len_language j->seq_len_vision k->head_dim
        # attention_score [batch, n_head, seq_len_language, seq_len_vision]

        if media_location:
            # TODO 若后续需要media location mask在此处添加 对attention_score进行mask
            pass

        attention_probs = tf.nn.softmax(attention_score)
        # attention_score经过softmax得到概率分布
        # attention_probs [batch, n_head, seq_len_language, seq_len_vision]

        output = tf.einsum('...ij,...jk->...ik', attention_probs, value)
        # i->seq_len_language j->seq_len_vision k->head_dim]
        # output [batch, n_head, seq_len_language, head_dim]
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, shape=(batch, seq_len_language, self.inner_dim))
        # output [batch, seq_len_language, inner_dim]
        output = self.output_layer(output)
        # output [batch, seq_len_language, dim]
        return output


class GatedCrossAttentionBlock(tf.keras.Model):
    """实现gated cross attention block 本质为gate + CrossAttentionWithMediaMask + gate + ffw"""
    def __init__(self, dim, n_head=8, head_dim=64, ffw_mult=4, attend_media_mask=False):
        """
        参数:
            dim: cross attention变量的feature维度
            n_head: multi-head attention的head个数
            head_dim: 每个head的维度
            ffw_mult: feedforward的参数量相对于feature的倍数 经验值为4
            attend_media_mask: 是否对media进行mask
        """
        super(GatedCrossAttentionBlock, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.ffw_mult = ffw_mult
        self.attend_media_mask = attend_media_mask
        self.cross_attention = CrossAttentionWithMediaMask(dim=self.dim, n_head=self.n_head, head_dim=self.head_dim,
                                                           attend_media_mask=self.attend_media_mask)
        self.attn_gate = tf.Variable(tf.constant([0.]))
        # attention的门控 仅用一个浮点数表示 注意必须初始化为0 使得初始时刻仅考虑文本信息
        self.ffw = utils.FeedForward(dim=self.dim, mult=self.ffw_mult)
        self.ffw_gate = tf.Variable(tf.constant([0.]))
        # feedforward的门控 仅用一个浮点数表示 注意必须初始化为0 使得初始时刻仅考虑文本信息

    def call(self, inputs, training=None, mask=None):
        """
        输入:
            visual_feature: 所有图像经过Perceiver_Resampler后得到的特征向量 [batch, seq_len_vision, dim]
            language_feature: 输入的文本信息特征 [batch, seq_len_language, dim]
            media_location: 图像对应的位置 [batch, seq_len_language] 默认是None
        输出:
            output: cross-attention的计算结果[batch, seq_len_language, dim]
        """
        visual_feature = inputs[0]
        language_feature = inputs[1]
        media_location = None
        if len(inputs) == 3:
            media_location = inputs[2]

        feature = language_feature + \
                  tf.math.tanh(self.attn_gate) * self.cross_attention((visual_feature, language_feature, media_location))
        # 将计算的cross_attention乘以tanh(attn_gate) 进行控制 并加上原始的language_feature
        feature = feature + \
                  tf.math.tanh(self.ffw_gate) * self.ffw(feature)
        # 经过ffw计算后的结果乘以tanh(ffw_gate) 进行控制 加上feature
        # feature的维度一直保持[batch, seq_len_language, dim]
        return feature


def main():
    batch = 32
    seq_len_vision = 64
    seq_len_language = 128
    dim = 128
    n_head = 8
    head_dim = 32
    ffw_mult = 4
    attn_block = GatedCrossAttentionBlock(dim=dim, n_head=n_head, head_dim=head_dim, ffw_mult=ffw_mult)
    visual_feature = tf.random.normal(shape=(batch, seq_len_vision, dim))
    language_feature = tf.random.normal(shape=(batch, seq_len_language, dim))
    output = attn_block(inputs=(visual_feature, language_feature))
    print(output)
    print(attn_block.summary())
    for layer in attn_block.layers:
        for weight in layer.weights:
            print(weight.name, weight.shape)


if __name__ == '__main__':
    main()

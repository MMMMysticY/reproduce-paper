import tensorflow as tf
import modeling
import utils
import pickle


# 本方法实现tf.keras.Model风格的广义attention + feedforward结构


class Attention(tf.keras.Model):
    """keras风格的广义attention"""
    def __init__(self, dim=768, n_head=12, head_dim=64, initializer_range=0.2):
        """
        参数:
            dim: cross attention变量的feature维度
            n_head: multi-head attention的head个数
            head_dim: 每个head的维度
            initializer_range: 全连接层参数初始化参数
        """
        super(Attention, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.initializer_range = initializer_range
        self.inner_dim = self.n_head * self.head_dim  # inner_dim是n_head * head_dim之后将要被reshape为n_head和head_dim
        self.scale = self.head_dim ** -0.5  # scale是1/根号下dk
        self.query_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            # use_bias=False,
            name='query',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
        )
        self.key_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            # use_bias=False,
            name='key',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
        )
        self.value_layer = tf.keras.layers.Dense(
            units=self.inner_dim,
            # use_bias=False,
            name='value',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
        )
        self.output_layer = tf.keras.layers.Dense(
            units=self.dim,
            # use_bias=False,
            name='output',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
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
            from_tensor 用于query的tensor [batch, seq_len_from, dim]
            to_tensor 用于key和value的tensor [batch, seq_len_to, dim]
            若from_tensor == to_tensor 即为self-attention 否则为cross-attention
            attention_mask 用于对attention加以mask [batch, seq_len_from, seq_len_to] 值为0或1 对0位置进行mask 对1位置进行保留
        输出:
            output 经过广义attention得到的结果 [batch, seq_len_from, n_head*head_dim]
        """
        from_tensor, to_tensor = inputs[:2]
        from_shape = modeling.get_shape_list(from_tensor, expected_rank=3)
        to_shape = modeling.get_shape_list(to_tensor, expected_rank=3)
        # assert from_shape[0] == to_shape[0]
        # assert from_shape[-1] == to_shape[-1] == self.dim
        batch = from_shape[0]
        seq_len_from = from_shape[1]
        seq_len_to = to_shape[1]
        # 计算维度

        query = self.query_layer(from_tensor)
        # query [batch, seq_len_from, inner_dim]
        key = self.key_layer(to_tensor)
        # key [batch, seq_len_to, inner_dim]
        value = self.value_layer(to_tensor)
        # value [batch, seq_len_to, inner_dim]

        query = self.transpose_for_scores(query, batch, self.n_head, seq_len_from, self.head_dim)
        key = self.transpose_for_scores(key, batch, self.n_head, seq_len_to, self.head_dim)
        value = self.transpose_for_scores(value, batch, self.n_head, seq_len_to, self.head_dim)
        # query [batch, n_head, seq_len_from, head_dim]
        # key value [batch, n_head, seq_len_to, head_dim]

        query = query * self.scale
        # query = query / 根号下dk

        attention_score = tf.einsum('...ik,...jk->...ij', query, key)
        # i->seq_len_from j->seq_len_to k->head_dim
        # attention_score [batch, n_head, seq_len_from, seq_len_to]

        if len(inputs) == 3:
            attention_mask = inputs[2]
            # [batch, seq_len_from, seq_len_to]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            # [batch, 1, seq_len_from, seq_len_to]
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            # 对于0进行mask对于1不mask 所以(1-mask_val) * -10000.0 0->-10000.0 1->0.0
            attention_score += adder
            # attention_score加上adder就是mask

        attention_probs = tf.nn.softmax(attention_score)
        # attention_score经过softmax得到概率分布
        # attention_probs [batch, n_head, seq_len_from, seq_len_to]

        output = tf.einsum('...ij,...jk->...ik', attention_probs, value)
        # i->seq_len_from j->seq_len_to k->head_dim
        # output [batch, n_head, seq_len_from, head_dim]

        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, shape=(batch, seq_len_from, self.inner_dim))
        output = self.output_layer(output)
        # [batch, seq_len_from, dim]
        return output


class LMBlock(tf.keras.Model):
    """LM Block 本质为self-attention + ffw"""
    def __init__(self, dim=768, n_head=12, ffw_mult=4):
        """
        参数:
            dim: cross attention变量的feature维度
            n_head: multi-head attention的head个数
            ffw_mult: feedforward的参数量相对于feature的倍数 经验值为4
            init_from_bert: 是否从预训练的BERT导入参数
            layer_number: 如果init_from_bert为True 即从BERT导入参数 layer_number代表导入BERT的层号
        """
        super(LMBlock, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.ffw_mult = ffw_mult
        if self.dim % n_head != 0:
            raise ValueError(
                "参数dim必须整除n_head"
            )
        self.head_dim = self.dim // self.n_head
        self.attn = Attention(dim=self.dim, n_head=self.n_head, head_dim=self.head_dim)
        self.norm = tf.keras.layers.LayerNormalization()
        self.ffw = utils.FeedForward(dim=self.dim, mult=self.ffw_mult)

    def call(self, inputs, training=None, mask=None):
        """
        输入:
            input_tensor: 输入的input_tensor [batch, seq_len, dim]
            目前LMBlock实现中 并没有mask 原因在于Flamingo所对应的场景不需要进行mask
        输出:
            output: 经过LM计算后的结果 [batch, seq_len, dim]
        """
        input_tensor = inputs
        input_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
        # assert self.dim == input_shape[2]

        input_tensor = input_tensor + self.attn(inputs=(input_tensor, input_tensor))
        # self_attention + residual
        input_tensor = self.norm(input_tensor)
        # attn之后layer norm
        input_tensor = input_tensor + self.ffw(inputs=input_tensor)
        # ffw + residual
        return input_tensor

    def load_bert_weights(self, layer_num=0):
        self.build(input_shape=(None, None, self.dim))
        attention_layer = self.get_layer(index=0)
        layer_norm = self.get_layer(index=1)
        ffw = self.get_layer(index=2)
        bert_chinese_params_file = open('all_bert_chinese_L-12_H-768_A-12_params.pkl', 'rb')
        bert_chinese_params = pickle.load(bert_chinese_params_file)
        bert_chinese_params_file.close()
        params = bert_chinese_params[layer_num]
        attention_layer.set_weights(params['attn'])
        layer_norm.set_weights(params['attn_norm'])
        ffw.set_weights(params['ffw'])


def main():
    batch = 32
    seq_len = 64
    dim = 768
    n_head = 12
    ffw_mult = 4
    lm_block = LMBlock(dim=dim,  n_head=n_head, ffw_mult=ffw_mult)
    # lm_block.build(input_shape=(None, None, dim))
    # print(lm_block.get_layer(index=0).get_weights())
    # print('---------------------------------------')
    # print(lm_block.get_layer(index=1).get_weights())
    # print('---------------------------------------')
    # print(lm_block.get_layer(index=2).get_weights())
    # print('---------------------------------------')
    lm_block.load_bert_weights(0)
    print(lm_block.get_layer(index=0).get_weights())
    print('---------------------------------------')
    print(lm_block.get_layer(index=1).get_weights())
    print('---------------------------------------')
    print(lm_block.get_layer(index=2).get_weights())
    print('---------------------------------------')
    # input_tensor = tf.random.normal(shape=(batch, seq_len, dim))
    # output = lm_block(input_tensor)
    # print(output)
    # print(lm_block.summary())


if __name__ == '__main__':
    main()

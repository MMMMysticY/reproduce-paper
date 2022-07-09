# encoding=utf-8
import tensorflow as tf
import modeling
import Gated_XATTN_Dense
import LM_Block
import Perceiver_Resampler
import Text_Embedding
import Vision_Encoder


class FusedTransformer(tf.keras.Model):
    def __init__(self, dim=768, n_head=8, head_dim=64, ffw_mult=4, initializer_range=0.2,  # 通用参数
                 img_token_id=3, seq_len_image=64, perceiver_num_layers=2, perceiver_num_latents=64,
                 img_preprocess=None,  # 图像参数
                 num_layers=6, cross_attention_every=1,  # 融合部分参数
                 vocab_size=21128, token_type_vocab_size=2, max_seq_len=512  # 文本参数
                 ):
        super(FusedTransformer, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.ffw_mult = ffw_mult
        self.initializer_range = initializer_range
        self.img_token_id = img_token_id
        self.seq_len_image = seq_len_image
        self.perceiver_num_layers = perceiver_num_layers
        self.perceiver_num_latents = perceiver_num_latents
        self.img_preprocess = img_preprocess
        self.num_layers = num_layers
        self.cross_attention_every = cross_attention_every
        self.vocab_size = vocab_size
        self.token_type_vocab_size = token_type_vocab_size
        self.max_seq_len = max_seq_len

        self.img_encoder = Vision_Encoder.VisionEncoder(dim=self.dim, seq_len_image=self.seq_len_image,
                                                        input_image=False)

        self.text_embedding = Text_Embedding.TextEmbedding(embedding_size=self.dim, vocab_size=self.vocab_size,
                                                           token_type_vocab_size=self.token_type_vocab_size,
                                                           max_position_embedding_size=self.max_seq_len)
        self.text_embedding.load_bert_weights()
        # text embedding对象
        # 导入bert的参数

        self.perceiver_resampler = Perceiver_Resampler.PerceiverResampler(dim=self.dim,
                                                                          num_layers=self.perceiver_num_layers,
                                                                          num_latent=self.perceiver_num_latents,
                                                                          n_head=self.n_head, head_dim=self.head_dim,
                                                                          ffw_mult=self.ffw_mult)
        # perceiver_resampler对象

        self.fused_transformer_layers = list()
        # 堆叠的融合层
        for index in range(self.num_layers):
            if index % cross_attention_every == 0:  # 每cross_attention_every个block加入一个Gated_XATTN_Dense
                self.fused_transformer_layers.append(
                    Gated_XATTN_Dense.GatedCrossAttentionBlock(dim=self.dim, n_head=self.n_head, head_dim=self.head_dim,
                                                               ffw_mult=self.ffw_mult))
            lm_block = LM_Block.LMBlock(dim=self.dim, n_head=self.n_head, ffw_mult=self.ffw_mult)
            lm_block.load_bert_weights(index)
            # 导入bert_chinese参数
            self.fused_transformer_layers.append(lm_block)
            # 每一层都有LM_Block

        self.output_proj = tf.keras.layers.Dense(
            units=self.vocab_size,
            use_bias=False,
            name='output_layer',
            # kernel_initializer=modeling.create_initializer(initializer_range)
            kernel_initializer=tf.keras.initializers.RandomNormal()
        )
        # 从dim->vocab_size进行token的预测

    def gather_indexes(self, output, masked_lm_positions):
        """
        输入：
            output: 输出文本的预测结果[batch, seq_len_language, vocab_size]
            masked_lm_positions: [batch, 1] 每个文本mask的位置 int
        输出:
            position_output: batch内每个位置对应预测位置的结果 [batch, vocab_size]
        """
        output_shape = modeling.get_shape_list(output, expected_rank=3)
        batch_size = output_shape[0]
        batch_size = tf.cast(batch_size, dtype=tf.int32)
        seq_len = output_shape[1]
        vocab_size = output_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_len, [-1, 1])
        # 得到每个batch的sequence绝对起始位置 以seq_len为区间进行分隔 -> [0, seq_len, seq_len*2, ... seq_len * batch-1]
        # [batch, 1]

        flat_positions = tf.reshape(masked_lm_positions + flat_offsets, [-1])
        # [batch, 1]的masked_lm_positions + [batch, 1]的flat_offsets 进行对应位置相加得到了每个预测位置的绝对位置[batch,1] -> [batch]
        flat_sequence_tensor = tf.reshape(output, [-1, vocab_size])
        # [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
        position_output = tf.gather(flat_sequence_tensor, flat_positions)
        # 使用gather方法 按照每个绝对位置去取tensor 取batch个 [batch x seq_len, vocab_size] -> [batch, vocab_size]
        return position_output

    def call(self, inputs, training=None, mask=None):
        """
        输入：
            input_ids: 输入文本的的id表示 [batch, seq_len_language]每个位置为文本的int32表示
            token_type_ids 表示token type的id [batch, seq_len_language]
            image: [batch, height, width, channel] 输入图像的[0, 255]表示 / [batch, image_embedding_dim]预先计算过embedding
            masked_lm_positions: [batch, 1] 每个文本mask的位置 int
        输出:
            output: 输出文本的预测结果[batch, vocab_size]
        """
        input_ids, token_type_ids, image, masked_lm_positions = inputs
        input_ids_shape = modeling.get_shape_list(input_ids, expected_rank=2)
        token_type_ids_shape = modeling.get_shape_list(token_type_ids, expected_rank=2)
        image_shape = modeling.get_shape_list(image, expected_rank=[2, 4])
        masked_lm_positions_shape = modeling.get_shape_list(masked_lm_positions, expected_rank=2)
        # assert input_ids_shape[-1] == token_type_ids_shape[-1]
        # assert input_ids_shape[0] == image_shape[0]

        text_embedding = self.text_embedding((input_ids, token_type_ids))
        # [batch, seq_len_language, dim]

        image_embedding = self.img_encoder(image)
        # [batch, seq_len_image, dim]
        image_embedding = self.perceiver_resampler(image_embedding)
        # [batch, seq_len_latent, dim]

        for layer in self.fused_transformer_layers:
            if isinstance(layer, LM_Block.LMBlock):
                layer.trainable = False
                text_embedding = layer(text_embedding)
            else:
                text_embedding = layer((image_embedding, text_embedding))

        output = self.output_proj(text_embedding)
        # [batch, seq_len_language, vocab_size]
        position_output = self.gather_indexes(output, masked_lm_positions)
        # [batch, vocab_size]
        return position_output


def main():
    model = FusedTransformer()
    input_ids = tf.constant(
        [[1, 32, 564, 2235, 525, 33, 2], [1, 53, 5346, 5345, 331, 313, 2], [643, 635, 42, 12, 53, 53, 2]])
    token_type_ids = tf.constant(
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]]
    )
    masked_lm_positions = tf.constant([[2], [3], [5]])
    image = tf.random.normal(shape=(3, 2048))
    # print('input_ids_shape: ', input_ids.shape)
    # print('token_type_ids_shape: ', token_type_ids.shape)
    # print('masked_lm_positions_shape: ', masked_lm_positions.shape)
    # print('image_shape: ', image.shape)
    output = model((input_ids, token_type_ids, image, masked_lm_positions))
    print(output)
    print(model.summary())


if __name__ == '__main__':
    main()

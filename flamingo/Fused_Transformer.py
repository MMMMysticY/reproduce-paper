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
                 img_token_id=3, seq_len_image=256, perceiver_num_layers=2, perceiver_num_latents=64, img_preprocess=None,  # 图像参数
                 num_layers=6, cross_attention_every=1,  # 融合部分参数
                 vocab_size=21128,  token_type_vocab_size=16, max_seq_len=512  # 文本参数
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

        self.img_encoder = Vision_Encoder.VisionEncoder(dim=self.dim, seq_len_image=self.seq_len_image)
        # self.img_encoder.trainable = False
        # 冻结img_encoder的参数

        self.text_embedding = Text_Embedding.TextEmbedding(embedding_size=self.dim, vocab_size=self.vocab_size,
                                                           token_type_vocab_size=self.token_type_vocab_size,
                                                           max_position_embedding_size=self.max_seq_len)
        # text embedding对象

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
            self.fused_transformer_layers.append(
                LM_Block.LMBlock(dim=self.dim, n_head=self.n_head, ffw_mult=self.ffw_mult))
            # 每一层都有LM_Block

        self.output_proj = tf.keras.layers.Dense(
            units=self.vocab_size,
            use_bias=False,
            name='output_layer',
            kernel_initializer=modeling.create_initializer(initializer_range)
        )
        # 从dim->vocab_size进行token的预测

    def call(self, inputs, training=None, mask=None):
        """
        输入：
            input_ids: 输入文本的的id表示 [batch, seq_len_language]每个位置为文本的int32表示
            token_type_ids 表示token type的id [batch, seq_len_language]
            image: [batch, height, width, channel] 输入图像的[0, 255]表示
        输出:
            output: 输出文本的预测结果[batch, seq_len_language, vocab_size]
        """
        input_ids, token_type_ids, image = inputs
        input_ids_shape = modeling.get_shape_list(input_ids, expected_rank=2)
        token_type_ids_shape = modeling.get_shape_list(token_type_ids, expected_rank=2)
        image_shape = modeling.get_shape_list(image, expected_rank=4)
        assert input_ids_shape == token_type_ids_shape
        assert input_ids_shape[0] == image_shape[0]

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

        return self.output_proj(text_embedding)

        
def main():
    batch = 3
    seq_len_language = 128
    height = 800
    width = 800
    channel = 3
    model = FusedTransformer()
    input_ids = tf.constant(
        [[1, 32, 564, 2235, 525, 33, 2], [1, 53, 5346, 5345, 331, 313, 2], [643, 635, 42, 12, 53, 53, 2]])
    token_type_ids = tf.constant(
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]]
    )
    image = tf.random.normal(shape=(batch, height, width, channel))
    output = model((input_ids, token_type_ids, image))
    print(output)


if __name__ == '__main__':
    main()

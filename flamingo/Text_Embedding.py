import tensorflow as tf
import modeling

# 若tf.get_variable出错 使用采用tf.Variable


class WordEmbedding(tf.keras.Model):
    """keras风格的word embedding"""

    def __init__(self, vocab_size, embedding_size, word_embedding_name='word_embeddings'):
        """
        参数:
            vocab_size: 词表大小
            embedding_size: 将文本token ids映射到的维度
            word_embedding_name: 用以构建embedding_table的名称
        """
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_embedding_name = word_embedding_name
        self.embedding_table = tf.get_variable(
            name=self.word_embedding_name,
            shape=[self.vocab_size, self.embedding_size],
            initializer=modeling.create_initializer())

    def call(self, inputs, training=None, mask=None):
        """
        输入：
            input_ids: 输入文本的的id表示 [batch, seq_len]每个位置为文本的int32表示
        输出:
            input_ids_embedding: 输入文本的word embedding后的结果 [batch, seq_len, embedding_size]
        """
        input_ids = inputs
        input_ids = tf.expand_dims(input_ids, axis=[-1])
        # 维度变为[batch, seq_len, 1]

        output = tf.nn.embedding_lookup(self.embedding_table, input_ids)
        # embedding后维度为[batch, seq_len, 1, embedding_size]
        input_shape = modeling.get_shape_list(input_ids, expected_rank=3)
        output = tf.reshape(output,
                            input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        # embedding维度变为[batch, seq_len, embedding_size]
        return output

    def get_embedding_table(self):
        return self.embedding_table

    def get_word_embedding_name(self):
        return self.word_embedding_name


class TokenTypeEmbedding(tf.keras.Model):
    """keras风格的token type embedding"""
    def __init__(self, token_type_vocab_size, embedding_size, token_embedding_name='token_type_embeddings'):
        """
        参数:
            token_type_vocab_size: token type的词表大小
            embedding_size: 将文本token ids映射到的维度
            token_embedding_name: 用以构建token_type_table的名称
        """
        super(TokenTypeEmbedding, self).__init__()
        self.token_type_vocab_size = token_type_vocab_size
        self.embedding_size = embedding_size
        self.token_embedding_name = token_embedding_name
        self.token_type_table = tf.get_variable(
            name=self.token_embedding_name,
            shape=[self.token_type_vocab_size, self.embedding_size],
            initializer=modeling.create_initializer()
        )

    def call(self, inputs, training=None, mask=None):
        """
        输入:
            input_tensor 经过word embedding后的tensor [batch, seq_len, embedding_size]
            token_type_ids 表示token type的id [batch, seq_len]
        输出:
            input_tensor + token_type_embedding [batch, seq_len, embedding_size]
        """
        input_tensor, token_type_ids = inputs
        input_tensor_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        # flat_token_type_ids [batch * seq_len]
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
        # one_hot_ids [batch * seq_len, token_type_vocab_size]
        token_type_embedding = tf.matmul(one_hot_ids, self.token_type_table)
        # token_type_embedding [batch * seq_len, embedding_size]
        token_type_embedding = tf.reshape(token_type_embedding, input_tensor_shape)
        # [batch, seq_len, embedding_size]
        return input_tensor + token_type_embedding

    def get_token_type_table(self):
        return self.token_type_table


class PositionEmbedding(tf.keras.Model):
    """keras风格的token type embedding"""
    def __init__(self, max_position_embedding_size, embedding_size, position_embedding_name='position_embedding'):
        """
        参数:
            max_position_embedding_size: position embedding的最大长度
            embedding_size: 将文本token ids映射到的维度
            position_embedding_name: 用以构建full_position_embedding的名称
        """
        super(PositionEmbedding, self).__init__()
        self.max_position_embedding_size = max_position_embedding_size
        self.embedding_size = embedding_size
        self.position_embedding_name = position_embedding_name
        self.full_position_embedding = tf.get_variable(
            name=self.position_embedding_name,
            shape=[self.max_position_embedding_size, self.embedding_size],
            initializer=modeling.create_initializer()
        )

    def call(self, inputs, training=None, mask=None):
        """
        输入:
            input_tensor 经过word embedding和token type embedding后的tensor [batch, seq_len, embedding_size]
        输出:
            input_tensor + position_embedidng [batch, seq_len, embedding_size]
        """
        input_tensor = inputs
        input_tensor_shape = modeling.get_shape_list(input_tensor)
        seq_len = input_tensor_shape[1]
        position_embedding = tf.slice(self.full_position_embedding, [0, 0], [seq_len, -1])
        # 截断到[seq_len, embedding_size]
        position_embedding = tf.reshape(position_embedding, [1, seq_len, self.embedding_size])
        # [1, seq_len, embedding_size]
        output = input_tensor + position_embedding
        return output

    def get_full_position_embedding(self):
        return self.full_position_embedding


class TextEmbedding(tf.keras.Model):
    """keras风格的word embedding + token type embedding + position embedding + layer_norm + dropout"""
    def __init__(self, embedding_size, vocab_size, token_type_vocab_size, max_position_embedding_size, dropout_p=0.1):
        """
        参数:
            embedding_size: 将文本token ids映射到的维度
            vocab_size: 词表大小
            token_type_vocab_size: token type的词表大小
            max_position_embedding_size: position embedding的最大长度
            dropout_p: dropout率
        """
        super(TextEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size= vocab_size
        self.token_type_vocab_size = token_type_vocab_size
        self.max_position_embedding_size = max_position_embedding_size
        self.dropout_p = dropout_p
        self.word_embedding_object = WordEmbedding(vocab_size=self.vocab_size, embedding_size=self.embedding_size)
        self.token_type_embedding_object = TokenTypeEmbedding(token_type_vocab_size=self.token_type_vocab_size, embedding_size=self.embedding_size)
        self.position_embedding_object = PositionEmbedding(max_position_embedding_size=self.max_position_embedding_size, embedding_size=self.embedding_size)

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_p)

    def call(self, inputs, training=None, mask=None):
        """
        输入：
            input_ids: 输入文本的的id表示 [batch, seq_len]每个位置为文本的int32表示
            token_type_ids 表示token type的id [batch, seq_len]
        输出:
            input_tensor: 输入文本的word embedding + token_type_embedding + position embedding
            后的结果 [batch, seq_len, embedding_size]
        """
        input_ids, token_type_ids = inputs
        output_word_embedding = self.word_embedding_object(input_ids)
        output_token_type_embedding = self.token_type_embedding_object(inputs=(output_word_embedding, token_type_ids))
        output_position_embedding = self.position_embedding_object(output_token_type_embedding)
        output = self.layer_norm(output_position_embedding)
        output = self.dropout(output)
        return output


def main():
    input_ids = tf.constant(
        [[1, 32, 564, 2235, 525, 33, 2], [1, 53, 5346, 5345, 331, 313, 2], [643, 635, 42, 12, 53, 53, 2]])
    token_type_ids = tf.constant(
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]]
    )
    text_embedding_object = TextEmbedding(embedding_size=768, vocab_size=21000, token_type_vocab_size=16, max_position_embedding_size=512)
    output = text_embedding_object(inputs=(input_ids, token_type_ids))
    print(output.shape)


if __name__ == '__main__':
    main()

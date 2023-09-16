import tensorflow as tf


class OrdinalClassfication:
    def __init__(self):
        self.dnn_hidden_units = [64, 32]
        self.class_num = 10  # 分类个数十个
        self.boundaries = [0, 1, 3, 6, 10, 15, 25, 36,
                           50, 66, 100]  # 十一个坐标划分为了分了十类 左闭右开区间为一类

        self.logits = None
        self.labels = None
        self.raw_labels = None
        self.predictions = None
        self.loss = None

    def build_output_graph(self, input_tensor):
        """
        模型网络就是全连接层 
        input_tensor: 特征embedding后拼接起来的tensor [batch, input_hidden]
        logits: [batch, class_num-1]
        """
        feature = self.dnn_net(input_tensor, self.dnn_hidden_units, 'oc')
        logits = self.logits_layer(feature, self.class_num-1, 'oc')
        # [batch, class_num-1]
        self.logits = logits

    def build_input_graph(self, labels):
        """
        从输入的true label处理为oc需要的输入
        labels [batch, 1] 为输入的true label
        """
        oc_label_list = []
        for index in range(len(self.boundaries-1)):
            if index < len(self.boundaries)-2:
                oc_label_list.append(
                    tf.cast(labels >= self.boundaries[index+1], tf.float32))
                # oc_label_list中为是否大于当前区间
        self.labels = tf.concat(oc_label_list, axis=-1)
        # [batch, class_num-1]
        self.raw_labels = labels
        # [batch, 1]

    def build_loss_graph(self):
        """
        损失函数是每个位置二分类的sigmoid交叉熵 即class_num-1个2分类任务
        """
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits
        ))

    def build_predict_graph(self):
        """
        预测的graph
        从logits生成每个位置的概率 argmax得到概率
        而后对于每个区间而言用中间值作为预测结果
        """
        pred_probs = self.generate_prob(self.logits)
        # [batch, class_num] 每个位置的概率值
        pred_indices = tf.argmax(pred_probs, axis=-1)
        # 得到预测结果
        values = []
        for index in range(len(self.boundaries)-1):
            values.append(tf.ones_like(
                self.raw_labels) * ((self.boundaries[index] + self.boundaries[index+1]) / 2.0))
            # [batch, 1]的预测值
        values_concat = tf.concat(values, axis=-1)
        # [batch, class_num]的预测值
        pred_mul = values_concat * \
            tf.one_hot(pred_indices, len(self.boundaries)-1)
        pred = tf.reduce_sum(pred_mul, axis=-1, keepdims=True)
        self.predictions = pred

    def generate_prob(self, raw_tensor):
        """
        实现：
            [1, class(1), class(2), ... class(N-1)]
                          - (减法)
            [class(1), class(2), ... class(N-1), 0]
        """
        # temp [batch, 1]只是为了获得[batch, 1]的样例tensor
        temp_tensor = self.raw_labels
        if raw_tensor is None:
            return tf.ones_like(temp_tensor, tf.float32)

        raw_tensor = tf.nn.sigmoid(raw_tensor)
        one_tensor = tf.ones_like(temp_tensor, tf.float32)
        zero_tensor = tf.zeros_like(temp_tensor, tf.float32)
        former_tensor = tf.concat([one_tensor, raw_tensor], axis=-1)
        latter_tensor = tf.concat([raw_tensor, zero_tensor], axis=-1)
        return former_tensor-latter_tensor

    def dnn_net(self, input_tensor, dnn_hidden_units, name):
        with tf.variable_scope(name_or_scope="{}_dnn_hidden_layer".format(name)) as scope:
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with tf.variable_scope("hidden_layer_{}".format(layer_id)) as dnn_layer_scope:
                    input_tensor = layers.fully_connected(
                        input_tensor,
                        num_hidden_units,
                        'lrelu',
                        scope=dnn_layer_scope,
                        normalizer_fn=layers.batch_norm
                    )
        return input_tensor

    def logits_layer(self, input_tensor=None, output_shape=1, name=None):
        with tf.variable_scope("{}_logits_layer".format(name)) as scope:
            logits = layers.linear(
                input_tensor,
                output_shape,
                scope=scope
            )
            bias = contrib_variables.model_variable(
                'bias_weight',
                shape=[output_shape],
                initializer=tf.zeros_initializer(),
                trainable=True
            )
            logits = logits + bias
        return logits

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.op import nn_ops
from tensorflow.contrib.framework.python.ops import variables as contrib_variables


# 简易版本 都很笼统
class MOMEModel:
    def build_mome_structure(self, input_tensor):
        expert_num = 2
        batch_size = 16
        n_tasks = 3
        # 2个expert 3个task

        dnn_hidden_units = [64, 32]
        gate_hidden_units = [16, 8, 2]
        # 隐层参数

        experts_list = []
        gates_list = []

        with tf.variable_scope(name_or_scope="mome") as scope:
            for num in range(expert_num):
                feature = self.dnn_net(
                    input_tensor, dnn_hidden_units, 'lrelu', "expert_{}".format(num))
                experts_list.append(feature)
                # expert_num个tensor 每个tensor为[batch, hidden]
            # 每个expert进行输入特征抽取

            for num in range(n_tasks):
                gate = self.dnn_net(
                    input_tensor, gate_hidden_units, 'softmax', "gate_{}".format(num))
                gates_list.append(gate)
                # n_tasks个tensor 每个tensor为[batch, expert_num]
            # 每个任务都对应一个gate的计算
            features = tf.concat(experts_list, axis=1)
            # [batch, expert_num, hidden]

            task_result_list = []
            # 方法一 用广播的方法 将每个expert的特征乘以对应系数 在expert_num维度求和 达到加权和的效果
            for num in range(n_tasks):
                gate = gates_list[num]  # [batch, expert_num]
                # [batch, expert_num, 1]
                gate = tf.expand_dims(gate, axis=-1)
                # 广播机制进行乘法 [batch, expert_num, hidden]
                gated_features = features * gate
                # 所有expert的结果加和得到结果 [batch, hidden]
                gated_features = tf.reduce_sum(gated_features, axis=1)
                task_result = self.logits_layer(
                    gated_features, 1, "task_{}".format(num))
                task_result_list.append(task_result)

            # 方法二 矩阵相乘法 直接做到了对应系数相乘和加和 本身矩阵相乘的功能就是这样
            for num in range(n_tasks):
                gate = gates_list[num]
                # [batch, expert_num]
                gate = tf.expand_dims(gate, axis=-1)
                # [batch, expert_num 1]
                gated_features = tf.matmul(features, gate, tanspose_a=True)
                # [batch, hidden, 1]
                gated_features = tf.reshape(
                    gated_features, (gated_features.shape[0], -1))
                # [batch, hidden]
                task_result = self.logits_layer(
                    gated_features, 1, "task_{}".format(num))
                task_result_list.append(task_result)

            return task_result

    def dnn_net(self, input_tensor, dnn_hidden_units, act_func, name):
        with tf.variable_scope(name_or_scope="{}_dnn_hidden_layer".format(name)) as scope:
            if act_func == 'lrelu':
                for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                    with tf.variable_scope("hidden_layer_{}".format(layer_id)) as dnn_layer_scope:
                        input_tensor = layers.fully_connected(
                            input_tensor,
                            num_hidden_units,
                            'lrelu',
                            scope=dnn_layer_scope,
                            normalizer_fn=layers.batch_norm
                        )
            else:
                for layer_id in range(dnn_hidden_units-1):
                    num_hidden_units = dnn_hidden_units[layer_id]
                    with tf.variable_scope("hidden_layer_{}".format(layer_id)) as dnn_layer_scope:
                        input_tensor = layers.fully_connected(
                            input_tensor,
                            num_hidden_units,
                            'lrelu',
                            scope=dnn_layer_scope,
                            normalizer_fn=layers.batch_norm
                        )
                    with tf.variable_scope('hidden_layer_{}'.format(len(dnn_hidden_units)-1)) as dnn_layer_scope:
                        input_tensor = layers.fully_conncted(
                            input_tensor,
                            num_hidden_units[-1],
                            'softmax'
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

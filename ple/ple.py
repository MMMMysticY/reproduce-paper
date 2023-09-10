import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.op import nn_ops
from tensorflow.contrib.framework.python.ops import variables as contrib_variables


# 简易版本 都很笼统
class PLEModel:
    def __init__(self):
        self.batch_size = 16
        self.n_tasks = 3                        # 3个task
        self.n_experts = [2, 3, 4]              # 3个task分别有2,3,4个expert
        self.share_expert_num = 3               # share网络的expert个数
        self.hidden_unit = [32, 32]             # 特征抽取部分的隐层神经元
        self.tower_hidden_unit = [32, 16, 8]    # tower部分的隐层神经元个数
        self.extraction_network_num = 3         # extraction网络结构的个数

    def build_ple_structure(self, input_feature):
        # input_feature [batch, hidden_sum] hidden_sum是所有输入特征embedding后的拼接总维度
        input_tensors = [input_feature] * (self.n_tasks+1)
        # 初始化时刻 每个位置的输入tensor都相同 就是input_embedding
        for index in range(self.extraction_network_num):
            input_tensors = self.build_extraction_network(input_tensors, index)
        result = self.build_tower(input_tensors)
        return result

    def build_extraction_network(self, input_tensors, extraction_network_index):
        """
        input_tensors: [E_1_tensor, E_2_tensor, ..., E_K_tensor, E_S_tensor] 
                        E_N_tensor: [batch, hidden]
        output_tensors: [E_1_tensor, E_2_tensor, ..., E_K_tensor, E_S_tensor] 
                        E_N_tensor: [batch, hidden]
        """
        output_tensors = []

        # 1. 计算每个任务的gate数值
        gate_list = []
        for task_index, task_expert_num in enumerate(self.n_tasks):
            output_shape = task_expert_num + self.share_expert_num
            input_tensor = input_tensors[task_index]
            input_tensor = self.dnn_net(input_tensor, self.hidden_unit, "EN{}_task{}_gate_dnn".format(
                extraction_network_index, task_index))
            gate = self.logits_layer(input_tensor, output_shape, 'softmax', "EN{}_task{}_gate_logits".formax(
                extraction_network_index, task_index))
            # [batch, s+k_expert_nums]
            gate_list.append(gate)

        # 2. 计算share网络的gate值
        share_gate_expert_num = sum(self.n_experts) + self.share_expert_num
        share_gate = self.dnn_net(
            input_tensors[-1], self.hidden_unit, "EN{}_share_gate_dnn".format(extraction_network_index))
        share_gate = self.logits_layer(
            input_tensor[-1], share_gate_expert_num, 'softmax', "EN{}_share_gate_logits".format(extraction_network_index))
        # [batch, all_expert_sum_nums]

        # 3. 计算share部分的特征
        share_features = []
        for expert_index in range(self.share_expert_num):
            feature = self.dnn_net(
                input_tensors[-1], self.hidden_unit, "EN{}_share_expert{}".format(expert_index))
            # [batch, hidden]
            share_features.append(feature)
            # share_expert_num个tensor 每个tensor [batch, hidden]

        # 4. 计算每个task部分的特征和结果
        all_features = []
        for task_index in range(self.n_tasks):
            task_expert_num = self.n_experts[task_index]
            input_tensor = input_tensors[task_index]
            task_features = []
            for expert_index in range(task_expert_num):
                feature = self.dnn_net(input_tensor, self.hidden_unit, "EN{}_task{}_expert{}".format(
                    extraction_network_index, task_index, expert_index))
                task_features.append(feature)
                # task_expert_num个tensor 每个tensor [batch, hidden]
            task_features = task_features + share_features
            # task_Feature和share_feature拼接起来 [batch, s+k_expert_num, hidden]
            task_feature = tf.concat(task_features, axis=1)
            # task_index的门控 [batch, s+k_expert_num]
            task_gate = gate_list[task_index]
            gated_feature = tf.matmul(task_feature, tf.expand_dims(
                task_gate, axis=-1), tanspose_a=True)  # 将gate值与特征做加权和 [batch, hidden, 1]
            gated_feature = tf.reshape(
                gated_feature, [-1, gated_feature.shape[1]])  # [batch, hidden]
            output_tensors.append(gated_feature)  # 即为task_index的输出结果

            all_features = all_features + task_features  # 将task特征添加到all_features中去

        # 5. 计算share部分的结果
        # all_expert_num个tensor [batch, hidden]
        all_features = all_features + share_features
        # [batch, all_expert_nums, hidden]
        all_feature = tf.concat(all_features, axis=1)
        gated_share_feature = tf.matmul(all_feature, tf.expand_dims(
            share_gate, axis=-1), tanspose_a=True)  # [batch, hidden, 1]
        gated_share_feature = tf.rehsape(
            gated_share_feature, [-1, gated_share_feature.shape[1]])
        output_tensors.append(gated_share_feature)
        return output_tensors

    def build_tower(self, input_tensors):
        """
        input_tensors: [E_1_tensor, E_2_tensor, ..., E_K_tensor, E_S_tensor] 
                        E_N_tensor: [batch, hidden]
        output_tensors: [E_1_tensor, E_2_tensor, ..., E_K_tensor, E_S_tensor] 
                        E_N_tensor: [batch, 1]
        """
        results = []
        for task_index in range(self.task_num):
            input_tensor = input_tensors[task_index]
            result = self.dnn_net(
                input_tensor, self.tower_hidden_unit, "task{}_tower_dnn".formax(task_index))
            result = self.logits_layer(
                input_tensor, 1, 'softmax', name="task{}_towner_logits".format(task_index))
            results.append(result)
        return results

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

    def logits_layer(self, input_tensor=None, output_shape=1, act_fun=None, name=None):
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
            if act_fun == 'softmax':
                logits = tf.nn.softmax(logits)
        return logits

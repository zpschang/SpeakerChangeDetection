import tensorflow as tf
from parameter import parameter
from tensorflow.contrib import rnn

class SpeakerClassifier():
    def __init__(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.feature = tf.placeholder(tf.float32, [None, parameter.block_size, parameter.feature_length])
        self.speaker = tf.placeholder(tf.int64, [None])
        flat = tf.reshape(self.feature, shape=[-1, parameter.block_size * parameter.feature_length])
        def lstm():
            def single_cell():
                return rnn.BasicLSTMCell(parameter.lstm_size)
            cell = rnn.MultiRNNCell([single_cell() for _ in range(parameter.num_layer)])

            output, state = tf.nn.dynamic_rnn(cell, self.feature, dtype=tf.float32)

            output = tf.reduce_max(output, axis=1)
            logits = tf.layers.dense(output, parameter.speaker_num)
            return logits
        def mlp(input):
            tensor = tf.layers.dense(input, 2000)
            tensor = tf.nn.relu(tensor)
            tensor = tf.layers.dense(tensor, 1000)
            tensor = tf.nn.relu(tensor)
            tensor = tf.layers.dense(tensor, parameter.speaker_num)
            return tensor
        def resnet(tensor):
            prev_tensor = tensor
            tensor = tf.layers.dense(tensor, 500)
            tensor = tf.nn.relu(tensor)
            tensor = tf.layers.dense(tensor, 500)
            tensor = tf.nn.relu(tensor) + prev_tensor
            return tensor

        # logits = tf.layers.dense(self.feature, parameter.speaker_num)
        """
        tensor = tf.layers.dense(flat, 500)
        tensor = tf.nn.relu(tensor)
        for _ in range(10):
            tensor = resnet(tensor)
        logits = tf.layers.dense(tensor, parameter.speaker_num)
        """
        # logits = mlp(flat)
        logits = lstm()
        self.soft_result = tf.nn.softmax(logits)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.speaker, logits=logits)
        self.loss = tf.reduce_sum(cross_entropy)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        # gradients, _ = tf.clip_by_global_norm(gradients, parameter.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(parameter.learning_rate)
        self.op_train = optimizer.apply_gradients(zip(gradients, params), global_step=self.global_step)
        self.result = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.speaker, self.result), tf.float32))


    def train(self, sess, batch):
        feature_list, speaker_list = batch
        """
        for feature, speaker in zip(feature_list, speaker_list):
            print 'feature:', feature.tolist()
            print 'speaker:', speaker
            raw_input()
        """
        feed_dict = {}
        feed_dict[self.feature] = feature_list
        feed_dict[self.speaker] = speaker_list
        result, loss, acc, _ = sess.run([self.result, self.loss, self.acc, self.op_train], feed_dict=feed_dict)
        for predict, real in zip(result, speaker_list):
            if predict == real:
                parameter.correct[real] += 1
            parameter.total[real] += 1
            parameter.predict[predict] += 1
        return loss, acc

    def evaluate(self, sess, batch):
        feature_list, speaker_list = batch
        feed_dict = {}
        feed_dict[self.feature] = feature_list
        result = sess.run(self.soft_result, feed_dict=feed_dict)

        return result
import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np
slim = tf.contrib.slim


class Model(object):
    def __init__(self):
        self.cardinality = 8
        self.res_block = 1
        self.batch_size = 256
        self.batch_size = 256
        self.amount = 12492
        self.step_per_epoch = self.amount // self.batch_size

        self.image = tf.placeholder(tf.float32, [None, 320, 320, 1], name='image')
        with tf.name_scope("label"):
            self.label = tf.placeholder(tf.int32, [None], name='label')
            self.one_hot = tf.one_hot(indices=self.label, depth=3, name='one_hot')

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.training = tf.placeholder(tf.bool)

        self.output = self.network(self.image)

        self.loss = self.get_loss(self.output, self.one_hot)

        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.one_hot, 1))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('train_accuracy', self.accuracy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.GradientDescentOptimizer(1E-5).minimize(self.loss, global_step=self.global_step)
            # learning_rate = tf.train.exponential_decay(1E-3, global_step=self.global_step, decay_steps=self.step_per_epoch, decay_rate=0.9)

        self.merged = tf.summary.merge_all()
        print("网络初始化成功")

    def conv_layer(self, input, filter, kernel, stride, padding='SAME', layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                       padding=padding)
            return network

    def global_average_pooling(self, x):
        return tf.reduce_mean(x, [1, 2])

    def average_pooling(self, x, pool_size=[2, 2], stride=2, padding='SAME'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = self.conv_layer(x, filter=16, kernel=[7, 7], stride=2, layer_name=scope + '_conv1')
            x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
            x = tf.nn.relu(x)
        return x

    def transform_layer(self, x, stride, depth, scope):
        with tf.name_scope(scope):
            x = self.conv_layer(x, filter=depth, kernel=[1, 1], stride=stride, layer_name=scope + '_conv1')
            x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
            x = tf.nn.relu(x)

            x = self.conv_layer(x, filter=depth, kernel=[3, 3], stride=1, layer_name=scope + '_conv2')
            x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
            x = tf.nn.relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = self.conv_layer(x, filter=out_dim, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
            x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            input_dim = int(np.shape(input_x)[-1])
            print(input_dim//2)
            layers_split = list()
            for i in range(self.cardinality):
                splits = self.transform_layer(input_x, stride=stride, depth=input_dim//2, scope=layer_name+'_splitN_'+str(i))
                layers_split.append(splits)
            return tf.concat(layers_split, axis=3)

    def residual_layer(self, input_x, out_dim, stride, layer_num, flag=True):
        for i in range(self.res_block):
            input_dim = int(np.shape(input_x)[-1])

            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_' + layer_num + '_' + str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i))
            if flag is True:
                input_x = self.average_pooling(input_x)
            channel = input_dim // 2
            pad_input_x = tf.pad(input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            input_x = tf.nn.relu(x + pad_input_x)
        return input_x

    def network(self, input_x):
        print(input_x)
        x = self.first_layer(input_x, scope='first_layer')
        print(x)
        x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
        x = self.residual_layer(x, out_dim=32, stride=1, layer_num='1', flag=False)
        print(x)
        x = self.residual_layer(x, out_dim=64, stride=2, layer_num='2', flag=True)
        print(x)
        x = self.residual_layer(x, out_dim=128, stride=2, layer_num='3', flag=True)
        print(x)
        x = self.residual_layer(x, out_dim=256, stride=2, layer_num='4', flag=True)
        print(x)
        x = self.global_average_pooling(x)
        print(x)
        x = tf.layers.dense(inputs=x, use_bias=False, units=3, name='linear')
        print(x)
        return x

    def get_loss(self, output_concat, onehot):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=output_concat, labels=onehot)
            loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss', loss)
        return loss

    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    def combined_static_and_dynamic_shape(self, tensor):
        static_tensor_shape = tensor.shape.as_list()
        dynamic_tensor_shape = tf.shape(tensor)
        combined_shape = []
        for index, dim in enumerate(static_tensor_shape):
            if dim is not None:
                combined_shape.append(dim)
            else:
                combined_shape.append(dynamic_tensor_shape[index])
        return combined_shape

    def convolutional_block_attention_module(self, feature_map, index, inner_units_ratio=0.5):
        with tf.variable_scope("cbam_%s" % (index)):
            feature_map_shape = self.combined_static_and_dynamic_shape(feature_map)
            # channel attention
            channel_avg_weights = tf.nn.avg_pool(value=feature_map, ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
                                                 strides=[1, 1, 1, 1], padding='VALID')
            channel_max_weights = tf.nn.max_pool(value=feature_map, ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
                                                 strides=[1, 1, 1, 1], padding='VALID')
            channel_avg_reshape = tf.reshape(channel_avg_weights, [feature_map_shape[0], 1, feature_map_shape[3]])
            channel_max_reshape = tf.reshape(channel_max_weights, [feature_map_shape[0], 1, feature_map_shape[3]])
            channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

            fc_1 = tf.layers.dense(inputs=channel_w_reshape, units=feature_map_shape[3] * inner_units_ratio,
                                   name="fc_1", activation=tf.nn.relu)
            fc_2 = tf.layers.dense(inputs=fc_1, units=feature_map_shape[3], name="fc_2", activation=None)
            channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
            channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
            channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
            feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
            # spatial attention
            channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
            channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

            channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling, shape=[feature_map_shape[0], feature_map_shape[1],
                                                  feature_map_shape[2], 1])
            channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling, shape=[feature_map_shape[0], feature_map_shape[1],
                                                  feature_map_shape[2], 1])

            channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
            spatial_attention = slim.conv2d(channel_wise_pooling, 1, [7, 7], padding='SAME',
                                            activation_fn=tf.nn.sigmoid, scope="spatial_attention_conv")
            feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
            return feature_map_with_attention



if __name__ == '__main__':
    model = Model()
    print(model.get_num_params())
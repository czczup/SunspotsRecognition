import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np


class Model(object):
    def __init__(self):
        self.cardinality = 8
        self.res_block = 1
        self.image = tf.placeholder(tf.float32, [None, 320, 320, 1], name='image')

        with tf.name_scope("label"):
            self.label = tf.placeholder(tf.int32, [None], name='label')
            self.one_hot = tf.one_hot(indices=self.label, depth=3, name='one_hot')

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.training = tf.placeholder(tf.bool)

        self.output = self.network(self.image)

        self.loss = self.get_loss(self.output, self.one_hot)

        self.batch_size = 256
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.one_hot, 1))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('train_accuracy', self.accuracy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(1e-5).minimize(self.loss, global_step=self.global_step)

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


if __name__ == '__main__':
    model = Model()
    print(model.get_num_params())
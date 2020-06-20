import tensorflow as tf
from functools import reduce
from operator import mul


class Model(object):
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, None, None, 1], name='image')

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

    def conv2d(self, x, input_filters, output_filters, kernel, strides=1, padding="SAME"):
        with tf.name_scope('conv'):
            shape = [kernel, kernel, input_filters, output_filters]
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
            return tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')

    def residual(self, x, num_filters, strides, with_shortcut=False):
        with tf.name_scope('residual'):
            conv1 = self.conv2d(x, num_filters[0], num_filters[1], kernel=1, strides=strides)
            bn1 = tf.layers.batch_normalization(conv1, axis=3, training=self.training)
            relu1 = tf.nn.relu(bn1)
            conv2 = self.conv2d(relu1, num_filters[1], num_filters[2], kernel=3)
            bn2 = tf.layers.batch_normalization(conv2, axis=3, training=self.training)
            relu2 = tf.nn.relu(bn2)
            conv3 = self.conv2d(relu2, num_filters[2], num_filters[3], kernel=1)
            bn3 = tf.layers.batch_normalization(conv3, axis=3, training=self.training)
            if with_shortcut:
                shortcut = self.conv2d(x, num_filters[0], num_filters[3], kernel=1, strides=strides)
                bn_shortcut = tf.layers.batch_normalization(shortcut, axis=3, training=self.training)
                residual = tf.nn.relu(bn_shortcut+bn3)
            else:
                residual = tf.nn.relu(x+bn3)
        return residual

    def network(self, image):
        with tf.name_scope("ResNet"):
            with tf.name_scope('stage1'):
                image = tf.nn.max_pool(image, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
                conv = self.conv2d(image, 1, 16, 7, 1)
                bn = tf.layers.batch_normalization(conv, axis=3, training=self.training)
                relu = tf.nn.relu(bn)
                pool = tf.nn.max_pool(relu, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
            with tf.name_scope('stage2'):
                res = self.residual(pool, [16, 8, 8, 32], 1, with_shortcut=True)
            with tf.name_scope('stage3'):
                res = self.residual(res, [32, 16, 16, 64], 2, with_shortcut=True)
            with tf.name_scope('stage4'):
                res = self.residual(res, [64, 32, 32, 128], 2, with_shortcut=True)
            with tf.name_scope('stage5'):
                res = self.residual(res, [128, 64, 64, 256], 2, with_shortcut=True)
            pool = self.global_max_pooling(res)
            output = tf.layers.dense(pool, units=3)
        return output

    def global_max_pooling(self, x):
        return tf.reduce_max(x, [1, 2])


    def global_mean_pooling(self, x):
        return tf.reduce_mean(x, [1, 2])


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
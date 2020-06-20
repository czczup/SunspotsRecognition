from model import Model
import tensorflow as tf
import time
import os


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature_description = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64)
    }
    temp  = tf.parse_single_example(serialized_example, feature_description)
    image = tf.decode_raw(temp['image'], tf.float32)
    width = tf.cast(temp['width'], tf.int64)
    height = tf.cast(temp['height'], tf.int64)

    image = tf.reshape(image, tf.convert_to_tensor([width, height, 1]))
    # image = tf.image.random_flip_left_right(image) # 随机左右翻转
    # image = tf.image.random_flip_up_down(image) # 随机上下翻转
    label = tf.cast(temp['label'], tf.int64)
    return image, label


def load_train_set():
    with tf.name_scope('input_train'):
        image_train, label_train = read_and_decode("dataset/tfrecord/train_con_processed_avgpool.tfrecord")
        image_batch_train, label_batch_train = tf.train.batch(
            [image_train, label_train], batch_size=batch_size, capacity=2048, num_threads=4, dynamic_pad=True
        )
    return image_batch_train, label_batch_train


def load_valid_set():
    with tf.name_scope('input_valid'):
        image_valid, label_valid = read_and_decode("dataset/tfrecord/valid_con_processed_avgpool.tfrecord")
        image_batch_valid, label_batch_valid = tf.train.batch(
            [image_valid, label_valid], batch_size=batch_size, capacity=2048, num_threads=4, dynamic_pad=True
        )
    return image_batch_valid, label_batch_valid


def train(model):
    # network
    amount = 19689
    image_batch_train, label_batch_train = load_train_set()
    image_batch_valid, label_batch_valid = load_valid_set()

    # Adaptive use of GPU memory.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # general setting
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # Recording training process.
        writer_train = tf.summary.FileWriter("models/"+dirId+"/logs/train", sess.graph)
        writer_valid = tf.summary.FileWriter("models/"+dirId+"/logs/valid", sess.graph)

        last_file = tf.train.latest_checkpoint("models/"+dirId)
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += [var for var in tf.global_variables() if "global_step" in var.name]
        var_list += tf.trainable_variables()
        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        if last_file:
            tf.logging.info('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)
        # train
        while True:
            time1 = time.time()
            image_train, label_train, step = sess.run(
                [image_batch_train, label_batch_train, model.global_step])
            _, loss_ = sess.run([model.optimizer, model.loss], feed_dict={model.image: image_train,
                                                                          model.label: label_train,
                                                                          model.training: True})
            print('[epoch %d, step %d/%d]: loss %.6f' % (
            step // (amount // batch_size), step % (amount // batch_size), amount // batch_size, loss_),
                  'time %.3fs' % (time.time() - time1))
            if step % 10 == 0:
                image_train, label_train = sess.run([image_batch_train, label_batch_train])
                acc_train, summary = sess.run([model.accuracy, model.merged], feed_dict={model.image: image_train,
                                                                                         model.label: label_train,
                                                                                         model.training: True})
                writer_train.add_summary(summary, step)
                image_valid, label_valid = sess.run([image_batch_valid, label_batch_valid])
                acc_valid, summary = sess.run([model.accuracy, model.merged], feed_dict={model.image: image_valid,
                                                                                         model.label: label_valid,
                                                                                         model.training: True})
                writer_valid.add_summary(summary, step)
                print('[epoch %d, step %d/%d]: train acc %.3f, valid acc %.3f' % (step // (amount // batch_size),
                                                                                  step % (amount // batch_size),
                                                                                  amount // batch_size, acc_train,
                                                                                  acc_valid),
                      'time %.3fs' % (time.time() - time1))
            if step % 100 == 0:
                print("Save the model Successfully")
                saver.save(sess, "models/"+dirId+"/model.ckpt", global_step=step)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    deviceId = input("device id: ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    dirId = input("dir id: ")
    model = Model()
    batch_size = model.batch_size
    train(model)
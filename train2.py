import tensorflow as tf
import os
from SpatialPyramidPooling import SpatialPyramidPooling
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


train_reader = tf.data.TFRecordDataset('dataset/tfrecord/train_con_vector_avgpool.tfrecord')
valid_reader = tf.data.TFRecordDataset('dataset/tfrecord/train_con_vector_avgpool.tfrecord')

feature_description = {
    'vector': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}


def _parse_function(exam_proto):
    temp = tf.io.parse_single_example(exam_proto, feature_description)
    vector = tf.io.decode_raw(temp['vector'], tf.float32)
    label = temp['label']
    vector = tf.reshape(vector, [1200, 2])
    var, gradient = tf.unstack(vector, axis=1)
    print(var, gradient)
    return gradient, label


BATCH_SIZE = 10240
DATASET_SIZE = 12492
EPOCHS = 200
train_dataset = train_reader.repeat(EPOCHS).shuffle(2560, reshuffle_each_iteration=True).map(_parse_function).batch(BATCH_SIZE)
valid_dataset = valid_reader.shuffle(2560, reshuffle_each_iteration=True).map(_parse_function).batch(BATCH_SIZE)


# base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
# base_model.trainable = False
# global_maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
# prediction_layer = tf.keras.layers.Dense(3)
# model = tf.keras.Sequential([
#     base_model,
#     global_maxpool_layer,
#     prediction_layer
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=(1200,)),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(3)
])
model.build(input_shape=(1200,))
model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer = tf.keras.optimizers.Adam(lr=1E-10),
            metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/mymodel-fc/", update_freq=1)

history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=DATASET_SIZE//BATCH_SIZE,
                    validation_data=valid_dataset, callbacks=[tensorboard_callback])
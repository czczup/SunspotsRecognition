import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

train_reader = tf.data.TFRecordDataset('dataset/tfrecord/train_avgpool.tfrecord')
valid_reader = tf.data.TFRecordDataset('dataset/tfrecord/valid_avgpool.tfrecord')

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_function(exam_proto):
    temp = tf.io.parse_single_example(exam_proto, feature_description)
    image = tf.io.decode_raw(temp['image'], tf.float32)
    label = temp['label']
    width, height = temp['width'], temp['height']
    image = tf.reshape(image, [width, height, 3])
    return image, label


BATCH_SIZE = 32
DATASET_SIZE = 19689
EPOCHS = 200
train_dataset = train_reader.repeat(EPOCHS).shuffle(2560, reshuffle_each_iteration=True).\
    map(_parse_function).padded_batch(BATCH_SIZE, padded_shapes=([None, None, 3], []))
valid_dataset = valid_reader.shuffle(2560, reshuffle_each_iteration=True).\
    map(_parse_function).padded_batch(BATCH_SIZE, padded_shapes=([None, None, 3], []))


base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)

base_model.trainable = False
global_maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(3)
# model = tf.keras.Sequential([
#     base_model,
#     global_maxpool_layer,
#     prediction_layer
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3)
])
model.build(input_shape=(None, None, None, 3))
model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer = tf.keras.optimizers.Adam(lr=1E-3),
            metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/mymodel-gmp-lr0.001/", update_freq=1)

history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=DATASET_SIZE//BATCH_SIZE,
                    validation_data=valid_dataset, callbacks=[tensorboard_callback])

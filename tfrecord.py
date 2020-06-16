import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
from astropy.io import fits
import os
import warnings
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
warnings.filterwarnings('ignore')

type2id = {"alpha": 0, "beta": 1, "betax": 2}

def get_data(path, sunspot_type):
    dir1 = path + "/magnetogram/" + sunspot_type
    dir2 = path + "/continuum/" + sunspot_type
    filenames = os.listdir(dir1)
    train_data, valid_data = [], []
    for filename in tqdm(filenames):
        file_path1 = dir1 + "/" + filename
        sunspot_id = int(filename.split(".")[2])
        hdul_magnetogram = fits.open(file_path1)
        hdul_magnetogram.verify('fix')
        data_magnetogram = hdul_magnetogram[1].data / 10000.0 + 0.5
        data_magnetogram = pool2d(data_magnetogram, 2, 2, 0, 'avg')
        data_magnetogram = data_magnetogram.astype(np.float32)
        file_path2 = dir2 + "/" + filename.replace("magnetogram", "continuum")
        hudl_continuum = fits.open(file_path2)
        hudl_continuum.verify('fix')
        data_continuum = hudl_continuum[1].data / 75000.0
        data_continuum = pool2d(data_continuum, 2, 2, 0, 'avg')
        data_continuum = data_continuum.astype(np.float32)
        data_sum = (data_continuum + data_magnetogram) / 2.0
        if sunspot_id >= 5800:
            valid_data.append([np.dstack((data_magnetogram, data_continuum, data_sum)), type2id[sunspot_type]])
        else:
            train_data.append([np.dstack((data_magnetogram, data_continuum, data_sum)), type2id[sunspot_type]])
    return train_data, valid_data


def generate_tfrecord(type, data, tfrecord_file):
    x = np.array([item[0] for item in data]).reshape(-1, 1)
    y = [item[1] for item in data]
    if type == "train":
        ros = RandomOverSampler(random_state=0)
        x, y = ros.fit_sample(x, y)
    x = x.reshape(-1)

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in tqdm(range(len(x))):
            image, label = x[i], y[i]
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    A = np.pad(A, padding, mode='constant')
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0], stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)


if __name__ == '__main__':
    train_alpha, valid_alpha = get_data(path="dataset/trainset", sunspot_type="alpha")
    print(len(train_alpha), len(valid_alpha))
    train_beta, valid_beta = get_data(path="dataset/trainset", sunspot_type="beta")
    print(len(train_beta), len(valid_beta))
    train_betax, valid_betax = get_data(path="dataset/trainset", sunspot_type="betax")
    print(len(train_betax), len(valid_betax))
    train_data = train_alpha + train_beta + train_betax
    valid_data = valid_alpha + valid_beta + valid_betax
    print(len(train_data), len(valid_data))
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)
    generate_tfrecord("train", train_data, tfrecord_file="dataset/tfrecord/train_avgpool.tfrecord")
    generate_tfrecord("valid", valid_data, tfrecord_file="dataset/tfrecord/valid_avgpool.tfrecord")




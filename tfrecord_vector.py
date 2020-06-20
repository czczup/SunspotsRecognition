import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
from astropy.io import fits
import os
import warnings
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from skimage.morphology import flood_fill
warnings.filterwarnings('ignore')

type2id = {"alpha": 0, "beta": 1, "betax": 2}


def get_data_vector(path, sunspot_type):
    dir = path + "/continuum/" + sunspot_type
    filenames = os.listdir(dir)
    train_data, valid_data = [], []
    for filename in tqdm(filenames):
        filepath = dir + "/" + filename
        sunspot_id = int(filename.split(".")[2])
        hdul = fits.open(filepath)
        hdul.verify('fix')
        data = hdul[1].data
        min_, mean_ = np.min(data), np.mean(data)
        data = - (np.clip(data, min_, mean_ - 2500) - min_) / (mean_ - 2500 - min_) + 1
        block1 = np.mean(data[0:10, 0:10])
        block2 = np.mean(data[-10:, 0:10])
        block3 = np.mean(data[-10:, -10:])
        block4 = np.mean(data[0:10, -10:])
        max_block_mean = np.max([block1, block2, block3, block4])
        if max_block_mean > 0.9:
            temp = np.where(data < 0.2, 0, 1)
            width, height = temp.shape
            if block1 == max_block_mean:
                temp = flood_fill(temp, (0, 0), 2)
            if block2 == max_block_mean:
                temp = flood_fill(temp, (width - 1, 0), 2)
            if block3 == max_block_mean:
                temp = flood_fill(temp, (width - 1, height - 1), 2)
            if block4 == max_block_mean:
                temp = flood_fill(temp, (0, height - 1), 2)
            mask = np.where(temp > 1.5, 0, 1)
            data = np.multiply(data, mask)
        var_ = np.var(data, axis=0)[0: 1200]
        gradient_ = np.gradient(var_)
        var_ = var_ * 100
        gradient_ = gradient_ * 100
        length = len(gradient_)
        left_pad = (1200 - length) // 2
        right_pad = 1200 - (1200 - length) // 2 - length
        var_ = np.pad(var_, (left_pad, right_pad), 'constant', constant_values=0).astype(np.float32)
        gradient_ = np.pad(gradient_, (left_pad, right_pad), 'constant', constant_values=0).astype(np.float32)
        if sunspot_id >= 5800:
            valid_data.append([np.stack([var_, gradient_], axis=1), type2id[sunspot_type]])
        else:
            train_data.append([np.stack([var_, gradient_], axis=1), type2id[sunspot_type]])
    return train_data, valid_data


def generate_tfrecord(type, data, tfrecord_file):
    x = [item[0] for item in data]
    y = [item[1] for item in data]

    # if type == "train":
    #     ros = RandomOverSampler(random_state=0)
    #     x, y = ros.fit_sample(x, y)
    # x = x.reshape(-1)

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in tqdm(range(len(x))):
            vector, label = x[i], y[i]
            feature = {
                'vector': tf.train.Feature(bytes_list=tf.train.BytesList(value=[vector.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    train_alpha, valid_alpha = get_data_vector(path="dataset/trainset", sunspot_type="alpha")
    print(len(train_alpha), len(valid_alpha))
    train_beta, valid_beta = get_data_vector(path="dataset/trainset", sunspot_type="beta")
    print(len(train_beta), len(valid_beta))
    train_betax, valid_betax = get_data_vector(path="dataset/trainset", sunspot_type="betax")
    print(len(train_betax), len(valid_betax))
    train_data = train_alpha + train_beta + train_betax
    valid_data = valid_alpha + valid_beta + valid_betax
    print(len(train_data), len(valid_data))
    np.random.shuffle(train_data)
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)
    np.random.shuffle(valid_data)
    generate_tfrecord("train", train_data, tfrecord_file="dataset/tfrecord/train_con_vector_avgpool.tfrecord")
    generate_tfrecord("valid", valid_data, tfrecord_file="dataset/tfrecord/valid_con_vector_avgpool.tfrecord")




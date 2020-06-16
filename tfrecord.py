import tensorflow as tf
import numpy as np
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
    filenames = os.listdir(dir1)[:100]
    train_data, valid_data = [], []
    for filename in tqdm(filenames):
        file_path1 = dir1 + "/" + filename
        sunspot_id = int(filename.split(".")[2])
        hdul_magnetogram = fits.open(file_path1)
        hdul_magnetogram.verify('fix')
        data_magnetogram = hdul_magnetogram[1].data
        data_magnetogram = data_magnetogram.astype(np.float32)
        file_path2 = dir2 + "/" + filename.replace("magnetogram", "continuum")
        hudl_continuum = fits.open(file_path2)
        hudl_continuum.verify('fix')
        data_continuum = hudl_continuum[1].data
        data_continuum = data_continuum.astype(np.float32)
        if sunspot_id >= 5800:
            valid_data.append([np.dstack((data_magnetogram, data_continuum)).tobytes(), type2id[sunspot_type]])
        else:
            train_data.append([np.dstack((data_magnetogram, data_continuum)), type2id[sunspot_type]])
    return train_data, valid_data


def generate_tfrecord(type, data, tfrecord_file):
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    print(len(x), len(y))
    if type == "train":
        ros = RandomOverSampler(random_state=0)
        x, y = ros.fit_sample(x, y)

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in tqdm(range(len(x))):
            image = x[i]
            label = y[i]
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

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
    generate_tfrecord("train", train_data, tfrecord_file="dataset/tfrecord/train.tfrecord")
    generate_tfrecord("valid", valid_data, tfrecord_file="dataset/tfrecord/valid.tfrecord")




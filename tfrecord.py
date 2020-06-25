import tensorflow as tf
import numpy as np
from astropy.io import fits
import os
import warnings
from tqdm import tqdm
from PIL import Image
import random
warnings.filterwarnings('ignore')

type2id = {"alpha": 0, "beta": 1, "betax": 2}
CONTINUUM_MAX_VALUE = 73367
CONTINUUM_MIN_VALUE = 206
MAGNETOGRAM_MAX_VALUE = 2000
MAGNETOGRAM_MIN_VALUE = -2000


def center_crop(image, x, y):
    width, height = image.size[0], image.size[1]
    crop_side = min(width, height)
    width_crop = (width-crop_side)//2
    height_crop = (height-crop_side)//2
    box = (width_crop, height_crop, width_crop+crop_side, height_crop+crop_side)
    image = image.crop(box)
    image = image.resize((x, y), Image.ANTIALIAS)
    return image


def get_data(path, sunspot_type, data_type, MIN_VALUE, MAX_VALUE):
    dir = path + "/" + data_type +"/" + sunspot_type
    filenames = os.listdir(dir)
    train_data, valid_data = [], []
    for filename in tqdm(filenames):
        filepath = dir + "/" + filename
        sunspot_id = int(filename.split(".")[2])
        hdul = fits.open(filepath)
        hdul.verify('fix')
        data = hdul[1].data
        data = np.clip(data, MIN_VALUE, MAX_VALUE)
        data = (data - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * 255
        data = data.astype(np.uint8)
        image = Image.fromarray(data)
        image = center_crop(image, 320, 320)
        data = np.array(image, dtype=np.float32) / 255.0
        if sunspot_id >= 5800:
            valid_data.append([data, type2id[sunspot_type]])
        else:
            train_data.append([data, type2id[sunspot_type]])
    return train_data, valid_data


def get_all_data(path, sunspot_type):
    dir = path + "/continuum/" + sunspot_type
    filenames = os.listdir(dir)
    train_data, valid_data = [], []
    for filename in tqdm(filenames):
        filepath1 = dir + "/" + filename
        sunspot_id = int(filename.split(".")[2])
        hdul1 = fits.open(filepath1)
        hdul1.verify('fix')
        data1 = hdul1[1].data
        data1 = np.clip(data1, CONTINUUM_MIN_VALUE, CONTINUUM_MAX_VALUE)
        data1 = (data1 - CONTINUUM_MIN_VALUE) / (CONTINUUM_MAX_VALUE - CONTINUUM_MIN_VALUE) * 255
        data1 = data1.astype(np.uint8)
        image1 = Image.fromarray(data1)
        image1 = center_crop(image1, 224, 224)
        data1 = np.array(image1, dtype=np.float32) / 255.0

        filepath2 = filepath1.replace("continuum", "magnetogram")
        hdul2 = fits.open(filepath2)
        hdul2.verify('fix')
        data2 = hdul2[1].data
        data2 = np.clip(data2, MAGNETOGRAM_MIN_VALUE, MAGNETOGRAM_MAX_VALUE)
        data2 = (data2 - MAGNETOGRAM_MIN_VALUE) / (MAGNETOGRAM_MAX_VALUE - MAGNETOGRAM_MIN_VALUE) * 255
        data2 = data2.astype(np.uint8)
        image2 = Image.fromarray(data2)
        image2 = center_crop(image2, 224, 224)
        data2 = np.array(image2, dtype=np.float32) / 255.0
        data_stack = np.stack([data1, data2], axis=-1)
        if sunspot_id >= 5800:
            valid_data.append([data_stack, type2id[sunspot_type]])
        else:
            train_data.append([data_stack, type2id[sunspot_type]])
    return train_data, valid_data


def generate_tfrecord(type, data, tfrecord_file):
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in tqdm(range(len(x))):
            image, label = x[i], y[i]
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


if __name__ == '__main__':

    # train_alpha, valid_alpha = get_all_data(path="dataset/trainset", sunspot_type="alpha")
    # print(len(train_alpha), len(valid_alpha))
    # train_beta, valid_beta = get_all_data(path="dataset/trainset", sunspot_type="beta")
    # print(len(train_beta), len(valid_beta))
    # train_betax, valid_betax = get_all_data(path="dataset/trainset", sunspot_type="betax")
    # print(len(train_betax), len(valid_betax))


    # train_alpha, valid_alpha = get_data(path="dataset/trainset", sunspot_type="alpha", data_type="magnetogram",
    #                                     MIN_VALUE=MAGNETOGRAM_MIN_VALUE, MAX_VALUE=MAGNETOGRAM_MAX_VALUE)
    # print(len(train_alpha), len(valid_alpha))
    # train_beta, valid_beta = get_data(path="dataset/trainset", sunspot_type="beta", data_type="magnetogram",
    #                                     MIN_VALUE=MAGNETOGRAM_MIN_VALUE, MAX_VALUE=MAGNETOGRAM_MAX_VALUE)
    # print(len(train_beta), len(valid_beta))
    # train_betax, valid_betax = get_data(path="dataset/trainset", sunspot_type="betax", data_type="magnetogram",
    #                                     MIN_VALUE=MAGNETOGRAM_MIN_VALUE, MAX_VALUE=MAGNETOGRAM_MAX_VALUE)
    # print(len(train_betax), len(valid_betax))


    train_alpha, valid_alpha = get_data(path="dataset/trainset", sunspot_type="alpha", data_type="continuum",
                                        MIN_VALUE=CONTINUUM_MIN_VALUE, MAX_VALUE=CONTINUUM_MAX_VALUE)
    print(len(train_alpha), len(valid_alpha))
    train_beta, valid_beta = get_data(path="dataset/trainset", sunspot_type="beta", data_type="continuum",
                                      MIN_VALUE=CONTINUUM_MIN_VALUE, MAX_VALUE=CONTINUUM_MAX_VALUE)
    print(len(train_beta), len(valid_beta))
    train_betax, valid_betax = get_data(path="dataset/trainset", sunspot_type="betax", data_type="continuum",
                                        MIN_VALUE=CONTINUUM_MIN_VALUE, MAX_VALUE=CONTINUUM_MAX_VALUE)
    print(len(train_betax), len(valid_betax))
    train_data = train_alpha + train_beta + train_betax
    valid_data = valid_alpha + valid_beta + valid_betax
    print(len(train_data), len(valid_data))
    np.random.shuffle(train_data)
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)
    np.random.shuffle(valid_data)
    generate_tfrecord("train", train_data, tfrecord_file="dataset/tfrecord/train_continuum_320.tfrecord")
    generate_tfrecord("valid", valid_data, tfrecord_file="dataset/tfrecord/valid_continuum_320.tfrecord")



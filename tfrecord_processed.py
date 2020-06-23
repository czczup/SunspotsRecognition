import tensorflow as tf
import numpy as np
from astropy.io import fits
import os
import warnings
from tqdm import tqdm
from PIL import Image
from skimage.morphology import flood_fill
warnings.filterwarnings('ignore')

type2id = {"alpha": 0, "beta": 1, "betax": 2}
CONTINUUM_MAX_VALUE = 73367
CONTINUUM_MIN_VALUE = 206


def center_crop(image, x, y):
    width, height = image.size[0], image.size[1]
    crop_side = min(width, height)
    width_crop = (width-crop_side)//2
    height_crop = (height-crop_side)//2
    box = (width_crop, height_crop, width_crop+crop_side, height_crop+crop_side)
    image = image.crop(box)
    image = image.resize((x, y), Image.ANTIALIAS)
    return image


def np2image(data, MIN_VALUE, MAX_VALUE):
    data = np.clip(data, MIN_VALUE, MAX_VALUE)
    data = (data - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * 255
    data = data.astype(np.uint8)
    image = Image.fromarray(data)
    return image


def process(data):
    min_, mean_ = np.min(data), np.mean(data)
    data_processed = - (np.clip(data, min_, mean_ - 2500) - min_) / (mean_ - 2500 - min_) + 1
    image = np2image(data_processed, 0, 1)
    return np.array(center_crop(image, 224, 224), dtype=np.float32)


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
        data_processed = process(data)
        if sunspot_id >= 5800:
            valid_data.append([data_processed, type2id[sunspot_type]])
        else:
            train_data.append([data_processed, type2id[sunspot_type]])
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
    generate_tfrecord("train", train_data, tfrecord_file="dataset/tfrecord/train_continuum_processed_224.tfrecord")
    generate_tfrecord("valid", valid_data, tfrecord_file="dataset/tfrecord/valid_continuum_processed_224.tfrecord")



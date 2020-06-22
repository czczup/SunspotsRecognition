import tensorflow as tf
import numpy as np
from astropy.io import fits
import os
import warnings
from tqdm import tqdm
from PIL import Image
from model import Model

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
    valid_data = []
    for filename in tqdm(filenames):
        filepath = dir + "/" + filename
        sunspot_id = int(filename.split(".")[2])
        if sunspot_id >= 5800:
            hdul = fits.open(filepath)
            hdul.verify('fix')
            data = hdul[1].data
            data = np.clip(data, MIN_VALUE, MAX_VALUE)
            data = (data - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * 255
            data = data.astype(np.uint8)
            image = Image.fromarray(data)
            image = center_crop(image, 224, 224)
            data = np.array(image, dtype=np.float32) / 255.0
            data = np.reshape(data, [224, 224, 1])
            valid_data.append([data, type2id[sunspot_type]])
    return valid_data


def get_all_data(path, sunspot_type):
    dir = path + "/continuum/" + sunspot_type
    filenames = os.listdir(dir)
    valid_data = []
    for filename in tqdm(filenames):
        filepath1 = dir + "/" + filename
        sunspot_id = int(filename.split(".")[2])
        if sunspot_id >= 5800:
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

            data_merge = np.stack([data1, data2], axis=-1)
            valid_data.append([data_merge, type2id[sunspot_type]])
    return valid_data




def test(data, model):
    data_new = [data[i:i + 256] for i in range(0, len(data), 256)]
    correct = 0
    outputs, softmaxs = [], []
    for temp in tqdm(data_new):
        x = [item[0] for item in temp]
        y = [item[1] for item in temp]
        output = sess.run(tf.argmax(model.output, 1), feed_dict={model.image: x, model.training: False})
        output = output.tolist()
        outputs += output
        softmax = sess.run(tf.nn.softmax(model.output), feed_dict={model.image: x, model.training: False})
        softmax = softmax.tolist()
        softmaxs += softmax
    for index, y_hat in enumerate(outputs):
        if y_hat == data[index][1]:
            correct += 1
    labels = [item[1] for item in data]
    f1_scores = f1_score(outputs, labels)
    print("accuracy: %.4f" % (correct / len(data)))
    print("beta f1 score: %.4f" % f1_scores[1])
    print("betax f1 score: %.4f" % f1_scores[2])
    print("alpha f1 score: %.4f" % f1_scores[0])
    return softmaxs


def f1_score(outputs, labels):
    def f1_target(A):
        H, M, F = 0, 0, 0
        for i in range(len(outputs)):
            output = outputs[i]
            label = labels[i]
            if label == A and output == A:
                H += 1
            elif label == A and output != A:
                M += 1
            elif label != A and output == A:
                F += 1
        recall = H / (H + M + 1E-7)
        precision = H / (H + F + 1E-7)
        f1 = 2 * recall * precision / (recall + precision + 1E-7)
        return f1
    return [f1_target(0), f1_target(1), f1_target(2)]


def load_model(dirId):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    last_file = tf.train.latest_checkpoint("models/" + dirId)
    var_list = [var for var in tf.global_variables() if "moving" in var.name]
    var_list += [var for var in tf.global_variables() if "global_step" in var.name]
    var_list += tf.trainable_variables()
    saver = tf.train.Saver(var_list=var_list)
    if last_file:
        tf.logging.info('Restoring model from {}'.format(last_file))
        saver.restore(sess, last_file)
    return sess

if __name__ == '__main__':
    deviceId = input("device id: ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    model = Model()
    sess = tf.Session()
    valid_alpha1 = get_data(path="dataset/trainset", sunspot_type="alpha", data_type="continuum",
                            MIN_VALUE=CONTINUUM_MIN_VALUE, MAX_VALUE=CONTINUUM_MAX_VALUE)
    valid_beta1 = get_data(path="dataset/trainset", sunspot_type="beta", data_type="continuum",
                           MIN_VALUE=CONTINUUM_MIN_VALUE, MAX_VALUE=CONTINUUM_MAX_VALUE)
    valid_betax1 = get_data(path="dataset/trainset", sunspot_type="betax", data_type="continuum",
                            MIN_VALUE=CONTINUUM_MIN_VALUE, MAX_VALUE=CONTINUUM_MAX_VALUE)
    valid_data1 = valid_alpha1 + valid_beta1 + valid_betax1

    valid_alpha2 = get_data(path="dataset/trainset", sunspot_type="alpha", data_type="magnetogram",
                            MIN_VALUE=MAGNETOGRAM_MIN_VALUE, MAX_VALUE=MAGNETOGRAM_MAX_VALUE)
    valid_beta2 = get_data(path="dataset/trainset", sunspot_type="beta", data_type="magnetogram",
                           MIN_VALUE=MAGNETOGRAM_MIN_VALUE, MAX_VALUE=MAGNETOGRAM_MAX_VALUE)
    valid_betax2 = get_data(path="dataset/trainset", sunspot_type="betax", data_type="magnetogram",
                            MIN_VALUE=MAGNETOGRAM_MIN_VALUE, MAX_VALUE=MAGNETOGRAM_MAX_VALUE)
    valid_data2 = valid_alpha2 + valid_beta2 + valid_betax2

    load_model("0002")
    softmax1 = test(valid_data1, model)
    load_model("0003")
    softmax2 = test(valid_data2, model)
    softmax = []
    correct = 0
    for i in range(1977):
        softmax.append((np.array(softmax1[i]+np.array(softmax2[i]))/2))
    outputs = np.argmax(softmax, 1)
    for index, y_hat in enumerate(outputs):
        if y_hat == valid_data1[index][1]:
            correct += 1
    labels = [item[1] for item in valid_data1]
    f1_scores = f1_score(outputs, labels)
    print("accuracy: %.4f" % (correct / len(valid_data1)))
    print("beta f1 score: %.4f" % f1_scores[1])
    print("betax f1 score: %.4f" % f1_scores[2])
    print("alpha f1 score: %.4f" % f1_scores[0])
import numpy as np
from astropy.io import fits
import os
import warnings
warnings.filterwarnings('ignore')


def data_transform(path, file_type, sunspot_type):
    dir = path + "/" + file_type + "/" + sunspot_type
    files = os.listdir(dir)
    if not os.path.exists(path+"_npy/"):
        os.makedirs(path+"_npy/")
    if not os.path.exists(path+"_npy/"+file_type):
        os.makedirs(path+"_npy/"+file_type)
    if not os.path.exists(path+"_npy/"+file_type+"/"+sunspot_type):
        os.makedirs(path+"_npy/"+file_type+"/"+sunspot_type)

    for filename in files:
        file_path = dir + "/" + filename
        hdul = fits.open(file_path)
        hdul.verify('fix')
        image_data = hdul[1].data
        print(image_data)
        save_path = path + "_npy/" + file_type + "/" + sunspot_type + "/"\
                    + "".join(filename.split(".")[:-1]) + ".npy"
        np.save(save_path, image_data.astype(np.float32))
    print("finish")


data_transform(path="trainset", file_type="magnetogram", sunspot_type="alpha")
data_transform(path="trainset", file_type="magnetogram", sunspot_type="beta")
data_transform(path="trainset", file_type="magnetogram", sunspot_type="betax")
data_transform(path="trainset", file_type="continuum", sunspot_type="alpha")
data_transform(path="trainset", file_type="continuum", sunspot_type="beta")
data_transform(path="trainset", file_type="continuum", sunspot_type="betax")

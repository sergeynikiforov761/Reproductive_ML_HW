import os

import imageio
import numpy as np
import yaml
from skimage.transform import resize


def load_train_set(dirname, dict_characters, img_size):
    x_train = []
    y_train = []
    for label, character in dict_characters.items():
        list_images = os.listdir(dirname + '/' + character)
        for image_name in list_images[0:150]:
            image = imageio.imread(dirname + '/' + character + '/' + image_name)
            x_train.append(resize(image, (img_size, img_size)))
            y_train.append(label)
    return np.array(x_train), np.array(y_train)


def load_test_set(dirname, dict_characters, img_size):
    x_test = []
    y_test = []
    for image_name in os.listdir(dirname):
        character_name = "_".join(image_name.split('_')[:-1])
        label = [label for label, character in dict_characters.items() if character == character_name][0]
        image = imageio.imread(dirname + '/' + image_name)
        x_test.append(resize(image, (img_size, img_size)))
        y_test.append(label)
    return np.array(x_test), np.array(y_test)


def load_yaml(config_path):
    return yaml.safe_load(open(config_path))

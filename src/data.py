import os

import imageio
import numpy as np
import yaml
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from skimage.transform import resize

dict_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
                   3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
                   7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard',
                   11: 'lisa_simpson',
                   12: 'marge_simpson', 13: 'mayor_quimby', 14: 'milhouse_van_houten', 15: 'moe_szyslak',
                   16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}


def load_train_set(dirname, img_size):
    x_train = []
    y_train = []
    for label, character in dict_characters.items():
        list_images = os.listdir(dirname + '/' + character)
        for image_name in list_images[0:150]:
            image = imageio.imread(dirname + '/' + character + '/' + image_name)
            x_train.append(resize(image, (img_size, img_size)))
            y_train.append(label)
    return np.array(x_train), np.array(y_train)


def load_test_set(dirname, img_size):
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


def generate_data(directory, img_size, num_classes, num_samples, is_train):
    # Split data for cross validation
    if is_train:
        x, y = load_train_set(directory, img_size)
    else:
        x, y = load_test_set(directory, img_size)
    # Reduce Sample Size for DeBugging
    x = x[0:num_samples]
    y = y[0:num_samples]
    # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    y = to_categorical(y, num_classes=num_classes)
    return x, y

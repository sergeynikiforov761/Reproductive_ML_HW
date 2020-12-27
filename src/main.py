import argparse

from data import *
from model import run_another_keras_augmented


def train_model(config_path):
    config = load_yaml(config_path)

    test_dir = config['test_set_dir']
    train_dir = config['train_set_dir']
    num_classes = config['num_claases']
    img_size = config['img_size']
    model_save_weights_dir = config['model_weights_save_dir']
    model_save_dir = config['model_save_dir']

    X_train, Y_train = generate_data(train_dir, img_size, num_classes, 3000, True)
    X_test, Y_test = generate_data(test_dir, img_size, num_classes, 1000, False)

    model = run_another_keras_augmented(X_train, Y_train, X_test, Y_test, dict_characters)
    model.save(model_save_dir)
    model.save_weights(model_save_weights_dir)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path', dest='config_path', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config_path)

from src.data import *
from src.model import run_another_keras_augmented

if __name__ == "__main__":
    config = load_yaml('../params.yaml')

    test_dir = '../' + config['test_set_dir']
    train_dir = '../' + config['train_set_dir']
    num_classes = config['num_claases']
    img_size = config['img_size']
    model_save_weights_dir = config['model_weights_save_dir']
    model_save_dir = config['model_save_dir']

    X_train, Y_train = generate_data(train_dir, img_size, num_classes, 3000, True)
    X_test, Y_test = generate_data(test_dir, img_size, num_classes, 1000, False)

    model = run_another_keras_augmented(X_train, Y_train, X_test, Y_test, dict_characters)
    model.save(model_save_dir)
    model.save_weights(model_save_weights_dir)

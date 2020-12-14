from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding

from src.data import *
from src.model import runAnotherKerasAugmented

if __name__ == "__main__":
    config = load_yaml('./params.yaml')

    test_dir = config['test_set_dir']
    train_dir = config['train_set_dir']
    num_classes = config['num_claases']
    img_size = config['img_size']

    dict_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
                       3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
                       7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard',
                       11: 'lisa_simpson',
                       12: 'marge_simpson', 13: 'mayor_quimby', 14: 'milhouse_van_houten', 15: 'moe_szyslak',
                       16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}

    # Split data for cross validation
    X_test, Y_test = load_test_set(test_dir, dict_characters, img_size)
    X_train, Y_train = load_train_set(train_dir, dict_characters, img_size)
    # Reduce Sample Size for DeBugging
    X_train = X_train[0:3000]
    Y_train = Y_train[0:3000]
    X_test = X_test[0:1000]
    Y_test = Y_test[0:1000]
    # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = to_categorical(Y_train, num_classes=num_classes)
    Y_test = to_categorical(Y_test, num_classes=num_classes)

    runAnotherKerasAugmented(X_train, Y_train, X_test, Y_test, dict_characters)

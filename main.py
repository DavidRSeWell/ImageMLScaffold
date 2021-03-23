import json
import numpy as np

from ml import data,utils
from ml.designer import Designer


def main(config_path):

    config = utils.load_config(config_path)

    print("CONFIG")
    print("----------------------------------------------------------")
    print(json.dumps(config, indent=4, sort_keys=True))

    designer = Designer(config_path)
    print("Loading in the data")

    designer.load_data()

    print("Data loaded showing stats")
    designer.image_data.get_data_stats()

    X = designer.transform_data()

    label_dict, y = designer.image_data.transform_labels()

    X_train, X_test, y_train, y_test = designer.create_train_data(X,y)

    print("Training size \n")
    print(X_train.shape)

    print("Testing size \n")
    print(X_test.shape)

    print("Training model \n")
    designer.train(X_train,y_train)

    print("Evaluating the model")
    print(designer.test(X_test,y_test))

    print("Done running main")


if __name__ == '__main__':

    config_path = "config.yaml"
    main(config_path)


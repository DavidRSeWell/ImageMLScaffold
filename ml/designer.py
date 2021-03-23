"""
The designer glues all the pieces together
"""
from pydoc import locate
from sklearn.model_selection import train_test_split

import numpy as np

from ml.data import ImageData
from ml.features import ImageFeature
from ml import utils

class Designer:
    """
    The designer will take in a model and
    """

    def __init__(self,config_path: str):
        self.image_data = ImageData.load_from_config(config_path)

        self._config_path = config_path
        self._config = utils.load_config(self._config_path)
        self._data = None
        self._model = self.load_model()

    def create_train_data(self, data: np.array, labels: np.array) -> tuple:
        X , y = data, labels
        train_config = self._config["train"]

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = train_config["split"], random_state = train_config["random_state"])

        return (X_train, X_test, y_train, y_test)

    def load_data(self) -> None:
        self._data = self.image_data.load_data()

    def transform_data(self) -> np.array:
        if self._data is None:
            print("No data to tranform \n")
            print("Load in some data first")
            return np.array([])

        orig_data = self._data.copy()
        features = self._config["data"]["features"]
        data = orig_data.copy()
        for key,val in features.items():
            if val:
                feature: ImageFeature = locate("ml.features." + str(key))()
                data: np.array = feature.transform_data(data)

        return data

    def load_model(self):
        model_type = self._config["model"]["type"]
        return locate("ml.model." + str(model_type))(self._config["model"]["args"])

    def train(self,X,y):
        self._model.train(X,y)

    def test(self,X,y) -> float:
        return self._model.test(X,y)


import numpy as np
import torch

from abc import abstractmethod
from sklearn.linear_model import LogisticRegression


class SKModel:
    def __init__(self):
        pass

    @abstractmethod
    def predict(self,X) -> np.array:
        pass

    @abstractmethod
    def train(self,X,y):
        pass

    @abstractmethod
    def test(self,X,y) -> float:
        pass


class LRModel(SKModel):

    def __init__(self,model_config):
        super(LRModel, self).__init__()
        self.model = LogisticRegression(**model_config)

    def predict(self,X) -> np.array:
        y = self.model.predict(X)
        return y

    def test(self,X,y) -> float:
        return self.model.score(X,y)

    def train(self,X,y):
        self.model.fit(X,y)
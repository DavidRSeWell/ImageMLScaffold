import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from ml import utils


class ImageData:

    @classmethod
    def load_from_config(cls,config_path: str):
        data = utils.get_data_info(config_path)
        return cls(data["path"],data["train"],data["test"],data["labels_path"])

    def __init__(self,data_dir: str,train_dir: str, test_dir: str, label_path: str):
        self._data_path = data_dir
        self._label_path = label_path
        self._train_path = self._data_path + train_dir
        self._test_path = self._data_path + test_dir

        self.labels = self.load_labels()

    def display_random_images(self, n: int = 5, dir: str = "train") -> list[np.array]:
        """
        Display n random images from the training dir
        :return:
        """
        num_images = len(self)

        images_idx = np.random.choice([i for i in range(1,num_images + 1)],size=n,replace=False)

        images = []

        for idx in images_idx:
            print(f"Image = {idx} label = {self.labels[idx - 1]}")
            img_name = f"{idx}.png"
            img = cv2.imread(self._train_path + "/" + img_name)
            images.append(img)
            plt.figure()
            plt.imshow(img)
            plt.show()

        return images

    def get_data_stats(self):

        assert len(self) == len(self.labels)

        counts = dict()
        for i in self.labels:
            counts[i] = counts.get(i, 0) + 1

        #tot = float(len(self))
        #for key in counts.keys():
        #    counts[key] = counts[key] / tot

        print("The distribution of labels")
        print(counts)

        print("Total number of data point")
        print(len(self))

    def load_labels(self) -> list[int]:
        labels = pd.read_csv(self._data_path + self._label_path)["label"].tolist()

        return labels

    def load_data(self) -> np.array:
        data = []
        for idx in range(1,len(self) + 1):
            img_name = f"{idx}.png"
            img = cv2.imread(self._train_path + "/" + img_name)
            data.append(img)

        return np.array(data)

    def show_image(self,image: np.array,title: str =""):

        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def transform_labels(self,) -> tuple:
        unique_labels = list(set(self.labels))
        label_dict = {unique_labels[v]: v for v in range(len(unique_labels))}
        new_labels = [label_dict[label] for label in self.labels]
        return label_dict, new_labels

    def __len__(self) -> int:
        n = 0
        for _ in os.listdir(self._train_path):
            n += 1
        return n



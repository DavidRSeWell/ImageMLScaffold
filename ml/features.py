import cv2
import numpy as np

from abc import abstractmethod


class ImageFeature:
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, img: np.array) -> np.array:
       pass

    @abstractmethod
    def transform_data(self,data: np.array) -> np.array:
        pass


class CV2Features(ImageFeature):

    def __init__(self):
        super().__init__()

    def transform(self,image, vector_size=32):
        try:
            # alg = cv2.KAZE_create()
            alg = cv2.SIFT_create()
            # Dinding image keypoints
            kps = alg.detect(image)
            # Getting first 32 of them.
            # Number of keypoints is varies depend on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            # computing descriptors vector
            kps, dsc = alg.compute(image, kps)
            if dsc is None:
                return np.random.random(vector_size * 64)
            # Flatten all of them in one big vector - our feature vector
            dsc = dsc.flatten()
            # Making descriptor of same size
            # Descriptor vector size is 64
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                # if we have less the 32 descriptors then just adding zeros at the
                # end of our feature vector
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

            if dsc.size > needed_size:

                dsc = dsc[:needed_size]
        except cv2.error as e:
            print('Error: ', e)
            return None

        return dsc

    def transform_data(self,data: np.array) -> np.array:
        """
        Assume data is in the form of n x i x j x k
        where n is the number of points in the data set
        :param data:
        :return:
        """
        n = data.shape[0]
        trans_data = []
        for i in range(n):
            trans_data.append(self.transform(data[i]))

        assert len(trans_data) == n

        return np.vstack(trans_data)

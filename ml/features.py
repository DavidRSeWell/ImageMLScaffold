import cv2
import numpy as np


class CV2Features:

    def __init__(self):
        pass

    def extract_features(self,image, vector_size=32):
        #assert type(image) ==
        try:
            # Using KAZE, cause SIFT, ORB and other was moved to additional module
            # which is adding addtional pain during install
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
                return np.random.random(64)
            # Flatten all of them in one big vector - our feature vector
            dsc = dsc.flatten()
            # Making descriptor of same size
            # Descriptor vector size is 64
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                # if we have less the 32 descriptors then just adding zeros at the
                # end of our feature vector
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            print('Error: ', e)
            return None

        return dsc
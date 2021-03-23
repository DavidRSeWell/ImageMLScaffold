import json
import numpy as np

from ml import data,utils


def main(config_path):

    config = utils.load_config(config_path)

    print("CONFIG")
    print("----------------------------------------------------------")
    print(json.dumps(config, indent=4, sort_keys=True))

    data_img: data.ImageData = data.ImageData.load_from_config(config_path)

    #data_img.display_random_images(n=5)

    labels = data_img.labels

    data_img.get_data_stats()

    print("Done running main")


if __name__ == '__main__':

    config_path = "config.yaml"
    main(config_path)


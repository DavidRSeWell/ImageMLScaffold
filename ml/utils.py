import os
import sys
import yaml



def get_data_info(config_path: str = "config.yaml") -> dict:
    """
    Gets the directory of the data of interest
    based on whats in the yaml configuration file
    :return:
    """

    path = os.getcwd() + "/" + config_path
    config = load_config(path)

    data = config["data"]

    print("Returning data")

    return data


def load_config(name: str) -> dict:

    try:
        with open(name) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    except:
        print(f"Unable to load config file {name}")
        print("Exiting program...")
        sys.exit(1)
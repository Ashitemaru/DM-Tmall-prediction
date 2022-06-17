import pandas as pd
import numpy as np

from config import config


def read_train_list():
	train_list = pd.read_csv(config["train_path"])
	return train_list




def read_test_list():
	test_list = pd.read_csv(config["test_path"])

	return test_list



if __name__ == "__main__":
    print("Use this module by import-ing it.")
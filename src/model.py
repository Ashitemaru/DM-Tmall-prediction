import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from config import config

def model_logistic(label_train, input_train):
	model = LogisticRegression(solver='liblinear')
	model.fit(input_train, label_train)
	return model


if __name__ == "__main__":
    print("Use this module by import-ing it.")
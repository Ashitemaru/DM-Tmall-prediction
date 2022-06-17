import pandas as pd
import numpy as np

from config import config

def read_user_info():
    user_info = pd.read_csv(config["user_info_path"])

    # 0.0 in "age_range" also stands for UNKNOWN, convert it to NaN
    # 2.0 in "gender" also stands for UNKNOWN, convert it to NaN
    user_info["age_range"].replace(0, np.nan, inplace = True)
    user_info["gender"].replace(2, np.nan, inplace = True)

    print("Successfully load user info:")
    print(user_info.info())

    return user_info

def read_user_log():
    user_log = pd.read_csv(config["user_log_path"])
    user_log = user_log.rename(columns = { "seller_id": "merchant_id" })
    
    return user_log

def read_train():
    train_df = pd.read_csv(config["train_path"])

    print("Successfully load train dataset:")
    print(train_df.info())

    return train_df

def read_test():
    test_df = pd.read_csv(config["test_path"])

    print("Successfully load train dataset:")
    print(test_df.info())

    return test_df

if __name__ == "__main__":
    print("Use this module by import-ing it.")
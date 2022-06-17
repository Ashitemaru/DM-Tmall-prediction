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
    """ This function will split the train set into train & validate.
    """
    train_df = pd.read_csv(config["train_path"])

    print("Successfully load train dataset:")
    print(train_df.info())

    # Separate to get train & validate
    tot_length = train_df.shape[0]
    ratio = int(config.get("train_validate_ratio", 4))
    validate_length = tot_length // (1 + ratio)
    train_length = tot_length - validate_length
    
    debug = config.get("debug", False)
    if debug:
        shuffled_train_df = train_df
    else:
        shuffled_train_df = train_df.sample(frac = 1).reset_index(drop = True)

    validate = shuffled_train_df.tail(validate_length)
    train = shuffled_train_df.head(train_length)
    print(f"Splitted data set: TRAIN = {train.shape[0]}, VALIDATE = {validate.shape[0]}")

    return train, validate

if __name__ == "__main__":
    print("Use this module by import-ing it.")
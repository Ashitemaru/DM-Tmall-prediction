from datetime import datetime
import pandas as pd
import numpy as np

from config import config

def read_user_info():
    # Silence! Witch!
    pd.options.mode.chained_assignment = None

    user_info = pd.read_csv(config["user_info_path"])

    # Reset all null to UNKNOWN
    user_info["age_range"][pd.isna(user_info["age_range"])] = 0
    user_info["gender"][pd.isna(user_info["gender"])] = 2
    user_info[["age_range", "gender"]] = user_info[["age_range", "gender"]].astype("uint8")

    print("Successfully load user info:")
    print(user_info.info())

    return user_info

def read_user_log():
    # Silence! Witch!
    pd.options.mode.chained_assignment = None

    user_log = pd.read_csv(config["user_log_path"])
    user_log = user_log.rename(columns = { "seller_id": "merchant_id" })


    # Fill all nulls
    user_log["brand_id"][pd.isna(user_log["brand_id"])] = 0
    user_log["brand_id"] = user_log["brand_id"].astype("uint16")


    # Split out month & day
    user_log["month"] = (user_log["time_stamp"] // 100).astype("uint8")
    user_log["day"] = (user_log["time_stamp"] % 100).astype("uint8")


    # Split out weekday
    def timestamp_to_weekday(x):
        return datetime.strptime(f"2016{x:04d}", "%Y%m%d").weekday()
    user_log["weekday"] = user_log["time_stamp"].apply(timestamp_to_weekday).astype("uint8")


    # Drop useless timestamp
    user_log = user_log.drop(["time_stamp"], axis = 1)

    print("Successfully load user log:")
    print(user_log.info())
    
    return user_log

def read_train():
    train_df = pd.read_csv(config["train_path"])

    print("Successfully load train dataset:")
    print(train_df.info())

    return train_df

def read_test():
    test_df = pd.read_csv(config["test_path"])

    print("Successfully load test dataset:")
    print(test_df.info())

    return test_df

def read_user_log_cache():
    user_log = pd.read_csv(config["user_log_cache_path"])

    print("Successfully load user log cache:")
    print(user_log.info())

    return user_log

def load_data():
    use_cache = config.get("use_cache", False)
    if use_cache:
        user_info = read_user_info()
        user_log = read_user_log_cache()
        train = read_train()
        test = read_test()
    else:
        user_info = read_user_info()
        user_log = read_user_log()
        train = read_train()
        test = read_test()

        user_log = pd.merge(user_log, train, how = "left", on = ["user_id", "merchant_id"])
        user_log["label"].fillna(-1, inplace = True)

        user_log.to_csv(config["user_log_cache_path"])

    return user_info, user_log, train, test

if __name__ == "__main__":
    print("Use this module by import-ing it.")
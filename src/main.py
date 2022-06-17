from preprocess import read_user_info, read_user_log, read_train
from analysis import analysis
from config import config
from train import train

def main():
    # Read in & preprocess
    user_info_df = read_user_info()
    print("Finish loading USER INFO")
    user_log_df = read_user_log()
    print("Finish loading USER LOG")

    mode = config.get("mode", "")
    if mode == "analysis": # Analysis
        analysis(user_info_df, user_log_df)

    elif mode == "train": # Train
        train_df, validate_df = read_train()
        train(train_df, validate_df, user_info_df, user_log_df)

    else:
        print("Invalid mode. Check the config.py")
        raise Exception()

if __name__ == "__main__":
    main()
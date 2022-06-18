from preprocess import read_user_info, read_user_log, read_train, read_test
from analysis import analysis
from config import config
from train import *

def main():
    # Read in & preprocess
    user_info_df = read_user_info()
    print("Finish loading USER INFO")
    user_log_df = read_user_log()
    print("Finish loading USER LOG")
    train_df = read_train()
    print("Finish loading TRAIN DATA")

    mode = config.get("mode", "")
    if mode == "analysis": # Analysis
        analysis(user_info_df, user_log_df, train_df)

    elif mode == "train": # Train
        test_df = read_test()
        print("Finish loading TEST DATA")

        train_df, test_df = dataset_preprocess(train_df, test_df, user_info_df, user_log_df)
        model_dict = train(train_df)
        
        if config["chosen_model_type"] not in model_dict:
            print("You have chosen a bad model type. Check the config.py")
        else:
            generate_answer(test_df, model_dict[config["chosen_model_type"]])

    else:
        print("Invalid mode. Check the config.py")
        raise Exception()

if __name__ == "__main__":
    main()
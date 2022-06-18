from preprocess import load_data
from analysis import analysis
from config import config
from train import *

def main():
    # Read in & preprocess
    user_info_df, user_log_df, train_df, test_df = load_data()

    mode = config.get("mode", "")
    if mode == "analysis": # Analysis
        analysis(user_info_df, user_log_df, train_df)

    elif mode == "train": # Train
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
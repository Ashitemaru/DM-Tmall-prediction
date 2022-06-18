config = {
    "mode": "train",

    "user_info_path": "../data/data_format1/user_info_format1.csv",
    "user_log_path": "../data/data_format1/user_log_format1.csv",
    "test_path": "../data/data_format1/test_format1.csv",
    "train_path": "../data/data_format1/train_format1.csv",
    "result_path": "../data/submission.csv",

    "debug": False,

    "chosen_model_type": "MLP",
    "model_types": ["Logistic", "MLP", "Decision-tree", "Random-forest", "Grad-tree", "Xgboost"],
}
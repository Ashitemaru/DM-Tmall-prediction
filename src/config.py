config = {
    "mode": "train",

    "user_info_path": "../data/data_format1/user_info_format1.csv",
    "user_log_path": "../data/data_format1/user_log_format1.csv",
    "user_log_cache_path": "../data/cache/user_log.csv",
    "test_path": "../data/data_format1/test_format1.csv",
    "train_path": "../data/data_format1/train_format1.csv",
    "result_path": "../data/submission.csv",

    "debug": True,
    "use_cache": True,

    "chosen_model_type": "Xgboost",
    "model_types": ["Grad-tree", "Xgboost"],
}
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

from feature import attatch_feature
from model import get_model
from config import config

def dataset_preprocess(train: pd.DataFrame, test: pd.DataFrame, user_info, user_log):
    # Silence! Witch!
    pd.options.mode.chained_assignment = None

    train_length = train.shape[0]
    test_length = test.shape[0]

    concatted_dataset = \
        pd.concat([
            train.drop(["label"], axis = 1),
            test.drop(["prob"], axis = 1)
        ]) \
            .reset_index() \
            .drop(["index"], axis = 1)

    featured_dataset, _, _, _ = attatch_feature(concatted_dataset, user_info, user_log)
    featured_train = featured_dataset.head(train_length)
    featured_test = featured_dataset.tail(test_length)

    featured_train["label"] = train["label"]
    featured_test["prob"] = test["prob"]

    return featured_train.reset_index(), featured_test.reset_index()

def train(train_data):
    # Split the model input & output
    label = train_data["label"]
    input_ = train_data.drop(["user_id", "merchant_id", "label"], axis = 1)

    # Split the train set & validate set
    label_train, label_validate, input_train, input_validate = train_test_split(
        label, input_,
        test_size = 0.2,
        random_state = 0
    )

    # Model dictionary
    model_dict = {}

    # Traverse model types
    for model_type in config["model_types"]:
        # Get the model
        model = get_model(model_type)

        # Train
        model.fit(input_train, label_train)

        # Evaluation on validate set
        label_validate_model = model.predict(input_validate)
        proba_validate_model = model.predict_proba(input_validate)
        acc = accuracy_score(label_validate, label_validate_model)
        ra_score = roc_auc_score(label_validate, proba_validate_model[:, 1])

        print(f"ACC score for model {model_type}: {acc}")
        print(f"ROC-AUC score for model {model_type}: {ra_score}")

        model_dict[model_type] = model

    return model_dict

def generate_answer(test_data, model):
    test_data["prob"] = model.predict_proba(
        test_data.drop(["user_id", "merchant_id", "prob"], axis = 1)
    )[:, 1]

    print(test_data.head(10))
    test_data[["user_id", "merchant_id", "prob"]].to_csv(config["result_path"], index = False)

if __name__ == "__main__":
    print("Use this module by import-ing it.")
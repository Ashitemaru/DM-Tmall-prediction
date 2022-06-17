from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from feature import attatch_feature
from model import get_model
from config import config

def train(train, user_info, user_log):
    train_data = attatch_feature(train, user_info, user_log)

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

def generate_answer(test, user_info, user_log, model):
    test_data = attatch_feature(test, user_info, user_log)
    
    test["prob"] = model.predict_proba(
        test_data.drop(["user_id", "merchant_id", "prob"], axis = 1)
    )[:, 1]

    print(test.head(10))
    test.to_csv(config["result_path"], index = False)

if __name__ == "__main__":
    print("Use this module by import-ing it.")
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import config
from model import *

def check(df, caption, length = 10):
    print("\n========== CHECK BEG ==========")
    print(caption)
    print(df.head(length))
    print("========== CHECK END ==========")

def attatch_feature(train, user_info, user_log):
    """ Attach features in user_info & user_log into the train dataset.
        Features including:
        1. User age range
        2. User gender
        3. Number of logs of this user in this merchant
        4. Number of categories of goods
        5. Browse days
        6. Number of clicks
        7. Number of shopping cart addings
        8. Number of purchasings
        9. Number of favourite addings
    """
    debug = config.get("debug", False)
    if debug:
        check(train, "Before feature-attaching")

    # Join user_info
    train = pd.merge(train, user_info, how = "left", on = "user_id")
    if debug:
        check(train, "After attaching user_info")

    def merge_col(base, index_key_list, rename_dict, agg = "count", remove_duplicated = True, df = user_log):
        """ This function will group the table merge the result into the base DF.

            @param base {pd.DataFrame}: The data frame to merge the result to.
            @param index_key_list {List[str]}: The list of keys to identify a row when grouping.
                Used both when operating rows & when merging results.
            @param rename_dict {Dict[str: str]}: The keys of this dict are columns to be operated,
                the values of this dict are the names which these columns to be renamed to.
            @param agg {str | lambda}: The aggregation function or description string.
            @param remove_duplicated {boolean}: Whether to remove duplicated rows.
                Default to True.
            @param df {pd.DataFrame}: The data frame to operate.
                In function 'attach_feature', default to 'user_log'

            @return {pd.DataFrame}: Merged data frame.
        """
        extended_key_list = index_key_list + list(rename_dict.keys())
        x = df[extended_key_list]

        if remove_duplicated: # Remove duplicated row
            x = x.groupby([x[k] for k in extended_key_list]).count()
            x = x.reset_index()

        # Group by given index key list & count the rows
        x = x.groupby([x[k] for k in index_key_list]).count()
        x = x.reset_index()

        # Rename the given column
        x = x.rename(columns = rename_dict)

        # Merge
        y = pd.merge(base, x, how = "left", on = index_key_list)
        if debug:
            check(y, f"After attaching {rename_dict[list(rename_dict.keys())[0]]}")

        return y

    # Constant
    UM_PAIR = ["user_id", "merchant_id"]

    # Join user_log
    train = merge_col(train, UM_PAIR, { "item_id": "log_num" }, False)
    train = merge_col(train, UM_PAIR, { "item_id": "item_num" })
    train = merge_col(train, UM_PAIR, { "cat_id": "category_num" })
    train = merge_col(train, UM_PAIR, { "time_stamp": "browse_days_num" })

    return train

def train(train, user_info, user_log):
    train_data = attatch_feature(train, user_info, user_log)
    print("Finish attatch feature")
    print(train_data.info())
    label = train_data["label"]
    input = train_data.drop(['user_id','merchant_id','label'],axis = 1)
    label_train, label_test, input_train, input_test = train_test_split(label, input, test_size=0.2, random_state=114514)
    return label_train, label_test, input_train, input_test
    
# attention the order of params
def launch_model_train(model, label_train, label_test, input_train, input_test):
    print("start train model:", model)

    # can add other models, in model.py
    if (model == "Logistic"):
        md = model_logistic(label_train, input_train)

    ans_label = md.predict(input_test) # return 0/1
    ans_prb = md.predict_proba(input_test) # return probability
    print(ans_label[0:100])

    ra_store = roc_auc_score(label_test, ans_prb[:, 1])
    print(ra_store)

    return md


if __name__ == "__main__":
    print("Use this module by import-ing it.")
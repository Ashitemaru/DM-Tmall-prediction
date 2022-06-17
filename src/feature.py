import pandas as pd

from config import config

def check(df, caption, length = 10):
    print("\n========== CHECK BEG ==========")
    print(caption)
    print(df.head(length))
    print("========== CHECK END ==========")

def attatch_feature(df, user_info, user_log):
    """ Attach features in user_info & user_log into the train dataset.
        Basic features including:
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
        check(df, "Before feature-attaching")

    # Join user_info
    df = pd.merge(df, user_info, how = "left", on = "user_id")
    if debug:
        check(df, "After attaching user_info")

    # Constant
    UM_PAIR = ["user_id", "merchant_id"]

    def merge_col(rename_dict, remove_duplicated = True, agg = "count", index_key_list = UM_PAIR):
        """ This function will group the user_log & merge the result into the train set.

            @param rename_dict {Dict[str: str]}: The keys of this dict are columns to be operated,
                the values of this dict are the names which these columns to be renamed to.
            @param agg {str | lambda}: The aggregation function or description string.
            @param remove_duplicated {boolean}: Whether to remove duplicated rows.
                Default to True.
            @param index_key_list {List[str]}: The list of keys to identify a row when grouping.
                Used both when operating rows & when merging results.

            @return {pd.DataFrame}: Merged data frame.
        """
        extended_key_list = index_key_list + list(rename_dict.keys())
        x = user_log[extended_key_list]

        if remove_duplicated: # Remove duplicated row
            x = x.groupby([x[k] for k in extended_key_list]).count()
            x = x.reset_index()

        # Group by given index key list & count the rows
        x = x.groupby([x[k] for k in index_key_list]).agg(agg)
        x = x.reset_index()

        # Rename the given column
        x = x.rename(columns = rename_dict)

        # Merge
        y = pd.merge(df, x, how = "left", on = index_key_list)
        if debug:
            check(y, f"After attaching {rename_dict[list(rename_dict.keys())[0]]}")

        return y

    # Join user_log
    df = merge_col({ "item_id": "log_num" }, False)
    df = merge_col({ "item_id": "item_num" })
    df = merge_col({ "cat_id": "category_num" })
    df = merge_col({ "time_stamp": "browse_days_num" })

    # Handle actions
    # Count out the times of action
    x = user_log[["user_id", "merchant_id", "action_type", "item_id"]]
    x = x.groupby([x["user_id"], x["merchant_id"], x["action_type"]]).count()
    x = x.reset_index()
    x = x.rename(columns = { "item_id": "count" })

    # Split into different cols according to action type
    x["click_num"] = (x["action_type"] == 0) * x["count"]
    x["cart_num"] = (x["action_type"] == 1) * x["count"]
    x["purchase_num"] = (x["action_type"] == 2) * x["count"]
    x["favourite_num"] = (x["action_type"] == 3) * x["count"]
    x = x.drop(["action_type", "count"], axis = 1)

    # Sum all up & merge
    x = x.groupby([x["user_id"], x["merchant_id"]]).sum()
    df = pd.merge(df, x, how = "left", on = UM_PAIR)
    if debug:
        check(df, "After attaching action")

    # Fill all the null
    df = df.fillna(method = "ffill")
    df = df.fillna(method = "bfill")

    print("Finish attaching features, info:")
    print(df.info())

    return df

if __name__ == "__main__":
    print("Use this module by import-ing it.")
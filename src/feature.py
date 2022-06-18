import pandas as pd

from config import config

def check(df, caption, length = 10):
    print("\n========== CHECK BEG ==========")
    print(caption)
    print(df.head(length))
    print("========== CHECK END ==========")

def attatch_feature(df, user_info, user_log: pd.DataFrame):
    # Constant
    USER = ["user_id"]
    MERCHANT = ["merchant_id"]
    UM_PAIR = ["user_id", "merchant_id"]
    debug = config.get("debug", False)

    # Feature tables
    user_feature = df[USER]
    merchant_feature = df[MERCHANT]
    union_feature = df[UM_PAIR]

    def create_feature(mode, rename_dict, remove_duplicated = True, agg = "count", src = user_log):
        """ This function will group the user_log & merge the result into the feature table.

            @param mode {int}: Which feature table to merge in:
                - 0 = USER
                - 1 = MERCHANT
                - 2 = UNION

            @param rename_dict {Dict[str: str]}: The keys of this dict are columns to be operated,
                the values of this dict are the names which these columns to be renamed to.

            @param agg {str | lambda}: The aggregation function or description string.

            @param remove_duplicated {boolean}: Whether to remove duplicated rows, default to True.

            @param src {pd.DataFrame} The data source, default to user_log

            @return {pd.DataFrame}: Merged data frame.
        """
        nonlocal user_feature, merchant_feature, union_feature
        
        if mode == 0:
            index_key_list = USER
        elif mode == 1:
            index_key_list = MERCHANT
        elif mode == 2:
            index_key_list = UM_PAIR
        else:
            print("Bad feature mode. Aborted!")
            raise Exception()

        extended_key_list = index_key_list + list(rename_dict.keys())
        x = src[extended_key_list]

        if remove_duplicated: # Remove duplicated row
            x = x.groupby([x[k] for k in extended_key_list]).count()
            x = x.reset_index()

        # Group by given index key list & count the rows
        x = x.groupby([x[k] for k in index_key_list]).agg(agg)
        x = x.reset_index()

        # Rename the given column
        x = x.rename(columns = rename_dict)

        # Merge
        caption = f"After attaching {' & '.join([rename_dict[k] for k in rename_dict])}"
        if mode == 0:
            y = pd.merge(user_feature, x, how = "left", on = index_key_list)
            if debug: check(y, caption)
            user_feature = y
        elif mode == 1:
            y = pd.merge(merchant_feature, x, how = "left", on = index_key_list)
            if debug: check(y, caption)
            merchant_feature = y
        elif mode == 2:
            y = pd.merge(union_feature, x, how = "left", on = index_key_list)
            if debug: check(y, caption)
            union_feature = y

    # User features
    create_feature(0, { "gender": "gender" }, False, "max", user_info)
    create_feature(0, { "age_range": "age_range" }, False, "min", user_info)

    # Double 11 features
    d11_user_log = user_log.filter(
        (user_log["month"] == 11) &
        user_log["day"].isin([10, 11, 12])
    )
    d11_user_purchase_log = d11_user_log.filter(
        d11_user_log["action_type"] == 2
    )

    # D11 feature #1 - item & brand & category unique count
    for mode, caption in zip([0, 1, 2], ["user", "merchant", "union"]):
        # All action types
        create_feature(mode, { "item_id": f"item_num_{caption}" }, True, "count", d11_user_log)
        create_feature(mode, { "brand_id": f"brand_num_{caption}" }, True, "count", d11_user_log)
        create_feature(mode, { "cat_id": f"category_num_{caption}" }, True, "count", d11_user_log)

        # Only purchase
        create_feature(mode, { "item_id": f"purchase_item_num_{caption}" }, True, "count", d11_user_purchase_log)
        create_feature(mode, { "brand_id": f"purchase_brand_num_{caption}" }, True, "count", d11_user_purchase_log)
        create_feature(mode, { "cat_id": f"purchase_category_num_{caption}" }, True, "count", d11_user_purchase_log)

    # Attach the feature
    df = pd.merge(df, user_feature, how = "left", on = USER)
    df = pd.merge(df, merchant_feature, how = "left", on = MERCHANT)
    df = pd.merge(df, union_feature, how = "left", on = UM_PAIR)

    return df

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
    y = x[["user_id", "merchant_id", "click_num", "purchase_num"]]
    x = x.groupby([x["user_id"], x["merchant_id"]]).sum()
    x["pur_click"] = x["purchase_num"] / (x["click_num"] + x["purchase_num"]) # 同一用户商家，购买点击比
    df = pd.merge(df, x, how = "left", on = UM_PAIR)
    tmp = y.groupby(y["user_id"]).sum()
    tmp["user_pur_click"] = tmp["purchase_num"] / (tmp["click_num"] + tmp["purchase_num"]) # 同一用户，购买点击比 
    tmp = tmp.drop(["merchant_id", "click_num", "purchase_num"], axis = 1)
    df = pd.merge(df, tmp, how = "left", on = "user_id")
    tmp = y.groupby(y["merchant_id"]).sum()
    tmp["merchant_pur_click"] = tmp["purchase_num"] / (tmp["click_num"] + tmp["purchase_num"]) # 同一商家，购买点击比
    tmp = tmp.drop(["user_id", "click_num", "purchase_num"], axis = 1)
    df = pd.merge(df, tmp, how = "left", on = "merchant_id")

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
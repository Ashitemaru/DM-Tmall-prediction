from venv import create
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

        # Group by given index key list
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

    def split_feature(mode, column, val_tgt_name_map, src = user_log):
        """ This function is similar to create_feature.
            Different params:

            @param column {str}: The column name to be counted.

            @param val_tgt_name_map {Dict[Any: str]}: The map from all the possible value of
                given column to the new column name.
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

        extended_key_list = index_key_list + [column]
        x = src[extended_key_list + ["item_id"]]

        # Group & count
        x = x.groupby([x[k] for k in extended_key_list]).count()
        x = x.reset_index()
        x = x.rename(columns = { "item_id": "count" })

        # Split
        for value in val_tgt_name_map:
            target_col_name = val_tgt_name_map[value]
            x[target_col_name] = (x[column] == value) * x["count"]
        x = x.drop(["count"], axis = 1)

        # Sum counters up
        x = x.groupby([x[k] for k in index_key_list]).sum()

        # Drop original column
        x = x.drop([column], axis = 1)

        # Merge
        caption = f"After attaching counters of {column}"
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

    # Figure
    if debug:
        check(user_log, "Check log ahead")
    user_log["date"] = user_log["month"] * 31 + user_log["day"]
    print("========== FIGURE BEG ==========")

    # User features
    create_feature(0, { "age_range": "age_range" }, False, "min", user_info)
    create_feature(0, { "gender": "gender" }, False, "max", user_info)

    # SPLIT START
    """
    for i in range(9):
        user_feature[f"age_{i}"] = 0
        user_feature[f"age_{i}"] = user_feature[f"age_{i}"].astype("uint8")
    for i in range(3):
        user_feature[f"gender_{i}"] = 0
        user_feature[f"gender_{i}"] = user_feature[f"gender_{i}"].astype("uint8")

    for i in range(0, user_feature.shape[0]):
        x = dict(user_feature.iloc[i])
        x[f"age_{user_feature.iloc[i]['age_range']}"] = 1
        x[f"gender_{user_feature.iloc[i]['gender']}"] = 1
        user_feature.iloc[i] = pd.Series(x)
    user_feature = user_feature.drop(["age_range", "gender"], axis = 1)
    """
    # SPLIT END

    # User figure 1
    create_feature(mode = 0, rename_dict = { "item_id": f"log_num" }, remove_duplicated = False)

    # User figure 3 - 6: merchant / item / brand / cat count
    create_feature(mode = 0, rename_dict = { "merchant_id": f"merchant_num" })
    create_feature(mode = 0, rename_dict = { "item_id": f"uitem_num" })
    create_feature(mode = 0, rename_dict = { "brand_id": f"ubrand_num" })
    create_feature(mode = 0, rename_dict = { "cat_id": f"ucat_num" })

    # User figure 7 - 10: action count
    uvtn_map = {
        0: f"uclick_num",
        1: f"ucart_num",
        2: f"upurchase_num",
        3: f"ufavorite_num",
    }
    split_feature(mode = 0, column = "action_type", val_tgt_name_map = uvtn_map, src = user_log)

    # User figure 11: user purchase / click 
    user_feature["upc_rate"] = \
        user_feature["upurchase_num"] / (user_feature["uclick_num"] + user_feature["upurchase_num"])

    # User figure 12: fisrt to last
    create_feature(mode = 0, rename_dict = { "date": f"ufirst_date" }, agg = "min")
    create_feature(mode = 0, rename_dict = { "date": f"ulast_date" }, agg = "max")
    user_feature["udis_date"] = (user_feature["ulast_date"] - user_feature["ufirst_date"]) * 24
    user_feature = user_feature.drop(["ufirst_date"], axis = 1)

    print("User feature OK!")

    # Merchant feature
    # Merchant figure 1
    create_feature(mode = 1, rename_dict = { "item_id": f"log_num" }, remove_duplicated = False)

    # Merchant figure 1 - 4: user / item / brand / cat count
    create_feature(mode = 1, rename_dict = { "user_id": f"user_num" })
    create_feature(mode = 1, rename_dict = { "item_id": f"mitem_num" })
    create_feature(mode = 1, rename_dict = { "brand_id": f"mbrand_num" })
    create_feature(mode = 1, rename_dict = { "cat_id": f"mcat_num" })

    # Merchant figure 5 - 8: action count
    mvtn_map = {
        0: f"mclick_num",
        1: f"mcart_num",
        2: f"mpurchase_num",
        3: f"mfavorite_num",
    }
    split_feature(mode = 1, column = "action_type", val_tgt_name_map = mvtn_map, src = user_log)

    # Merchant figure 9: merchant purchase / click
    merchant_feature["mpc_rate"] = \
        merchant_feature["mpurchase_num"] / (merchant_feature["mclick_num"] + merchant_feature["mpurchase_num"])

    # Merchant figure 10: rebuy
    mrebuy_map = {
        0: f"new_user",
        1: f"dup_user",
        -1: f"out_user",
    }
    split_feature(mode = 1, column = "label", val_tgt_name_map = mrebuy_map, src = user_log)
    merchant_feature = merchant_feature.drop(["new_user", "out_user"], axis = 1)

    print("Merchant feature OK!")

    # User-Merchant feature
    # User-Merchant figure 1 - 3: item / brand / cat count
    create_feature(mode = 2, rename_dict = { "item_id": f"umitem_num" })
    create_feature(mode = 2, rename_dict = { "brand_id": f"umbrand_num" })
    create_feature(mode = 2, rename_dict = { "cat_id": f"umcat_num" })

    # User-Merchant figure 4 - 7:
    umvtn_map = {
        0: f"umclick_num",
        1: f"umcart_num",
        2: f"umpurchase_num",
        3: f"umfavorite_num",
    }
    split_feature(mode = 2, column = "action_type", val_tgt_name_map = umvtn_map, src = user_log)

    # User-Merchant figure 8:
    union_feature["umpc_rate"] = \
        union_feature["umpurchase_num"] / (union_feature["umpurchase_num"] + union_feature["umclick_num"])

    # User-Merchant figure 9: fisrt to last
    create_feature(mode = 2, rename_dict = { "date": f"umfirst_date" }, agg = "min")
    create_feature(mode = 2, rename_dict = { "date": f"umlast_date" }, agg = "max")
    union_feature["umdis_date"] = (union_feature["umlast_date"] - union_feature["umfirst_date"]) * 24
    union_feature = union_feature.drop(["umfirst_date"], axis = 1)

    print("Union feature OK!")
    print("========== FIGURE END ==========")

    # Attach the feature
    df = pd.merge(df, user_feature.drop_duplicates(), how = "left", on = USER)
    df = pd.merge(df, merchant_feature.drop_duplicates(), how = "left", on = MERCHANT)
    df = pd.merge(df, union_feature.drop_duplicates(), how = "left", on = UM_PAIR)

    print("Finish attaching features, info:")
    print(df.info())

    return df, user_feature, merchant_feature, union_feature

if __name__ == "__main__":
    print("Use this module by import-ing it.")
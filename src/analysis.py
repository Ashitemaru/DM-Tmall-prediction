from matplotlib import use
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from feature import attatch_feature

def analysis_user_info(user_info):
    def get_cnt(x, key):
        if x is None:
            return user_info.isna().sum()[key]
        else:
            return user_info[user_info[key] == x][key].count()

    age_range_y = np.array(
        [get_cnt(None, "age_range")] +
        [get_cnt(i + 1, "age_range") for i in range(6)] +
        [get_cnt(7, "age_range") + get_cnt(8, "age_range")]
    )
    gender_y = np.array([
        get_cnt(None, "gender"),
        get_cnt(0, "gender"),
        get_cnt(1, "gender")
    ])

    age_range_x = np.array([
        "null", "under 18", "18~24", "25~29",
        "30~34", "35~39", "40~49", "over 50"
    ])
    gender_x = np.array(["unknown", "female", "male"])

    # Age hist
    plt.figure()
    plt.bar(age_range_x, age_range_y, label = "Person count", color = 'b')
    plt.legend()
    plt.title("User age range distribution")
    plt.savefig("../image/user_info_age_range.png")

    # Gender hist
    plt.figure()
    plt.bar(gender_x, gender_y, label = "Person count", color = 'b')
    plt.legend()
    plt.title("User gender distribution")
    plt.savefig("../image/user_info_gender.png")

    # Age and Gender
    plt.figure()
    colors = ['#0000FF', '#00FF00', '#FF0000']
    sns.countplot(x='age_range', order = [0, 1, 2, 3, 4, 5, 6, 7, 8], hue='gender', data = user_info, palette=colors)
    plt.legend()
    plt.title("User age-gender distribution")
    plt.savefig("../image/user_info_age-gender.png")


def analysis_user_log(user_log, df_train):
    def get_cnt(x, key):
        if x is None:
            return user_log.isna().sum()[key]
        else:
            return user_log[user_log[key] == x][key].count()

    # Log Info
    # Time stamp hist
    time_bins = np.array([
        510, 611, 711, 811, 911, 1011, 1115
    ])
    plt.figure()
    plt.hist(user_log["time_stamp"], bins = time_bins, label = "Time stamp", color = 'b')
    plt.grid(alpha = 0.5)
    plt.title("Log time histogram")
    plt.savefig("../image/log_time.png")

    # Log action hist
    action_y = np.array([
        get_cnt(0, "action_type"),
        get_cnt(1, "action_type"),
        get_cnt(2, "action_type"),
        get_cnt(3, "action_type")
    ])
    action_x = np.array(["click", "cart", "purchase", "favorite"])

    plt.figure()
    plt.bar(action_x, action_y, label = "Log count", color = 'b')
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.title("Log action distribution")
    plt.savefig("../image/log_action.png")

    
    # User-Merchant log num
    log_num_temp = user_log.groupby([user_log["user_id"], user_log["merchant_id"]]).count().reset_index()[["user_id", "merchant_id", "item_id"]]
    log_num_temp.rename(columns={"item_id":"log_num"}, inplace = True)
    df_analysis = pd.merge(df_train, log_num_temp, on = ["user_id", "merchant_id"], how = "left")
    print(log_num_temp.head(10))
    plt.figure()
    df_analysis["log_num"].hist(range = [0, 100], bins = 100, color = "b")
    #plt.hist(user_log["user_id"].value_counts(), range = [0, 100], bins = 100, label = "Log num", color="b")
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.title("Log num histogram")
    plt.savefig("../image/user-merchant_log_num.png")  

def analysis_features(df_feature):
    #User-Merchant purchase-click rate
    plt.figure()
    df_feature["umpc_rate"].hist(range = [0, 1], bins = 20, color = "b")
    df_feature[df_feature['label']==1]["umpc_rate"].hist(range = [0, 1], bins = 20, color = "r")
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.title("User-merchant num histogram")
    plt.savefig("../image/user-merchant_pc_rate.png") 

    #User-Merchant click num
    plt.figure()
    df_feature["umclick_num"].hist(range = [0, 60], bins = 60, color = "b")
    df_feature[df_feature['label']==1]["umclick_num"].hist(range = [0, 60], bins = 60, color = "r")
    plt.grid(alpha = 0.5)
    #plt.legend()
    plt.title("User-merchant num histogram")
    plt.savefig("../image/user-merchant_click_num.png")

    #User-Merchant cart num
    plt.figure()
    df_feature["umcart_num"].hist(range = [0, 5], bins = 10, color = "b")
    df_feature[df_feature['label']==1]["umcart_num"].hist(range = [0, 5], bins = 10, color = "r")
    plt.grid(alpha = 0.5)
    #plt.legend()
    plt.title("User-merchant num histogram")
    plt.savefig("../image/user-merchant_click_num.png")

    #User-Merchant purchase num
    plt.figure()
    df_feature["umpurchase_num"].hist(range = [0, 10], bins = 10, color = "b")
    df_feature[df_feature['label']==1]["umpurchase_num"].hist(range = [0, 10], bins = 10, color = "r")
    plt.grid(alpha = 0.5)
    #plt.legend()
    plt.title("User-merchant num histogram")
    plt.savefig("../image/user-merchant_purchase_num.png") 

    #User-Merchant favorite num
    plt.figure()
    df_feature["umfavorite_num"].hist(range = [0, 10], bins = 10, color = "b")
    df_feature[df_feature['label']==1]["umfavorite_num"].hist(range = [0, 10], bins = 10, color = "r")
    plt.grid(alpha = 0.5)
    #plt.legend()
    plt.title("User-merchant num histogram")
    plt.savefig("../image/user-merchant_favorite_num.png") 

    #User-Merchant dis date
    plt.figure()
    df_feature["umdis_date"].hist(range = [0, 180], bins = 18, color = "b")
    df_feature[df_feature['label']==1]["umdis_date"].hist(range = [0, 180], bins = 18, color = "r")
    plt.grid(alpha = 0.5)
    #plt.legend()
    plt.title("User-merchant num histogram")
    plt.savefig("../image/user-merchant_distance_date.png") 

    # Heatmap
    plt.figure(figsize=(30,24))
    col = df_feature.columns.tolist()[2:]
    mcorr = df_feature[col].corr()
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)]=True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    g = sns.heatmap(mcorr, cmap=cmap, mask=mask, square=True, annot=True, fmt='0.2f')
    plt.savefig("../image/user-merchant_heatmap.png") 

def analysis(user_info, user_log, train):
    analysis_user_info(user_info)
    analysis_user_log(user_log, train)

    featured_train, _, _, _ = attatch_feature(train, user_info, user_log)
    analysis_features(featured_train)
    # TODO: Draw images

if __name__ == "__main__":
    print("Use this module by import-ing it.")
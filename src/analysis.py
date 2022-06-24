import numpy as np
import matplotlib.pyplot as plt

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

    # Union distribution
    # TODO

def analysis_user_log(user_log):
    def get_cnt(x, key):
        if x is None:
            return user_log.isna().sum()[key]
        else:
            return user_log[user_log[key] == x][key].count()
    
    # User Info
    # User log num
    plt.figure()
    print(type(user_log["user_id"].value_counts().head(10)))
    print((user_log["user_id"].value_counts()).head(10))
    user_log["user_id"].value_counts().hist(range = [0, 15000], bins = 100, label = "Log num", color="b")
    #plt.hist(user_log["user_id"].value_counts(), range = [0, 100], bins = 100, label = "Log num", color="b")
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.title("Log num histogram")
    plt.savefig("../image/user_log_num.png")
     

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


def analysis(user_info, user_log, train):
    analysis_user_info(user_info)
    analysis_user_log(user_log)

    #featured_train = attatch_feature(train, user_info, user_log)
    # TODO: Draw images

if __name__ == "__main__":
    print("Use this module by import-ing it.")
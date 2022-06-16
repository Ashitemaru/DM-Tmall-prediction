from preprocess import read_user_info, read_user_log
from analysis import analysis

def main():
    # Read in & preprocess
    user_info_df = read_user_info()
    user_log_df = read_user_log()

    # Analysis
    analysis(user_info_df, user_log_df)

if __name__ == "__main__":
    main()
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from ..utils.logging_setup import logging

def prepare_sets(df: pd.DataFrame, train_size: float, random_state: int) -> pd.DataFrame:
    """
    Prepare train, validation and test sets from the given dataframe.
    :param df: pd.DataFrame: DataFrame with the data
    :param test_size: float: Size of the test set
    :param random_state: int: Random state for the split
    :return: pd.DataFrame, pd.DataFrame, pd.DataFrame: Train, validation and test sets
    """
    df.dropna(inplace=True)

    train_df, temp_df = train_test_split(df, test_size= ( 1 - train_size ), random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

    return train_df, val_df, test_df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: python sets_preparations.py <path_to_your_data.csv> <save_dir>")
        exit(1)

    CSV_PATH = sys.argv[1]
    SAVE_DIR = sys.argv[2] + "/splits/"

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logging.error(f"Error while reading the data: {e}")
        exit(1)

    train_df, val_df, test_df = prepare_sets(df, train_size=0.8, random_state=42)

    train_df.to_csv(f"{SAVE_DIR}train.csv", index=False)
    val_df.to_csv(f"{SAVE_DIR}val.csv", index=False)
    test_df.to_csv(f"{SAVE_DIR}test.csv", index=False)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

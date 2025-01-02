import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from ..utils.logging_setup import logging

def load_sets(path: str) -> pd.DataFrame:
    """
    Load the data from the given path.
    :param path: str: Path to the data
    :return: pd.DataFrame: DataFrame with the data
    """
    try:
        train_df = pd.read_csv(f"{path}/train.csv")
        val_df = pd.read_csv(f"{path}/val.csv")
        test_df = pd.read_csv(f"{path}/test.csv")
    except Exception as e:
        logging.error(f"Error while reading the data: {e}")
        exit(1)

    return train_df, val_df, test_df 

def prepare_sets(df: pd.DataFrame, train_size: float, random_state: int, limit:int = 0) -> pd.DataFrame:
    """
    Prepare train, validation and test sets from the given dataframe.
    :param df: pd.DataFrame: DataFrame with the data
    :param train_size: float: Size of the test set
    :param random_state: int: Random state for the split
    :return: pd.DataFrame, pd.DataFrame, pd.DataFrame: Train, validation and test sets
    """
    df.dropna(inplace=True)

    if limit != 0:
        df = df.sample(limit, random_state=random_state)

    train_df, temp_df = train_test_split(df, test_size= ( 1 - train_size ), random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

    return train_df, val_df, test_df

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logging.error("Usage: python sets_preparations.py <path_to_your_data.csv> <save_dir>")
        exit(1)

    CSV_PATH = sys.argv[1]
    SAVE_DIR = sys.argv[2] + "/splits/"
    LIMIT = int(sys.argv[3])

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logging.error(f"Error while reading the data: {e}")
        exit(1)

    train_df, val_df, test_df = prepare_sets(df, train_size=0.8, random_state=42, limit=LIMIT)

    train_df.to_csv(f"{SAVE_DIR}train.csv", index=False)
    val_df.to_csv(f"{SAVE_DIR}val.csv", index=False)
    test_df.to_csv(f"{SAVE_DIR}test.csv", index=False)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

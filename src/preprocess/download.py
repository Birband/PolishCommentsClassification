from datasets import load_dataset
import sys
from ..utils.logging_setup import logger

def load_and_save_dataset(name : str, dir : str) -> None:
    """
    Downloads the dataset from HuggingFace and saves it to a CSV file.
    """
    try:
        dataset = load_dataset(name)
    except:
        logger.error(f"Dataset {name} not found")
        sys.exit(1)
    
    logger.info(f"Loaded dataset {name}")

    for split in dataset.keys():
        dataset[split].to_csv(f"{dir}/civil_comments_{split}.csv")

    logger.info(f"Saved dataset to {dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: python download.py <dataset_name> <save_dir>")
        sys.exit(1)

    dataset = sys.argv[1]
    save_dir = sys.argv[2]

    load_and_save_dataset(dataset, save_dir)



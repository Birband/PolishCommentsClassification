help:
	@echo "download - Download the dataset"
	@echo "translate - Translate the dataset"
	@echo "split - Split the dataset into train, validation and test sets"

default:
	help

SAVE_DIR="data"
DATASET_NAME="google/civil_comments"
TRANSLATED_CSV="data/translated/translated_civil_comments_google.csv"
LIMIT?=0

download:
	python -m src.preprocess.download $(DATASET_NAME) $(SAVE_DIR)

translate:
	python -m src.preprocess.translate

split:
	python -m src.preprocess.sets_preparations $(TRANSLATED_CSV) $(SAVE_DIR) $(LIMIT)

train:
	python -m src.train.main
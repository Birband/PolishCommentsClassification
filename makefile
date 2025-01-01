default:
	help

help:
	@echo "download - Download the dataset"

SAVE_DIR="data"
DATASET_NAME="google/civil_comments"
MODEL_NAME="SnypzZz/Llama2-13b-Language-translate"

download:
	python -m src.preprocess.download $(DATASET_NAME) $(SAVE_DIR)

translate:
	python -m src.preprocess.translate
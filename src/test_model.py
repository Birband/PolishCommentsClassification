from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import f1_score
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

MIN_ERROR = 0.05
MAX_LEN = 512

class2id = {
    "toxicity": 0,
    "severe_toxicity": 1,
    "obscene": 2,
    "threat": 3,
    "insult": 4,
    "identity_attack": 5,
    "sexual_explicit": 6
}
id2class = {v: k for k, v in class2id.items()}

class ToxicCommentsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, target_columns=["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_columns = target_columns

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row["text"]
        labels = row[self.target_columns].values.astype(float)

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

def eval_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

model_name = "models/large_pl_28.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=7)

model.load_state_dict(torch.load(model_name, device), device)

CSV_FILE = "data/test/.csv"
df = pd.read_csv(CSV_FILE, encoding="utf-8")
dataset = ToxicCommentsDataset(df, tokenizer, MAX_LEN)
loader = DataLoader(dataset, batch_size=8)
model.to(device)

test_lables, test_preds = eval_model(model, loader, device)
xd = test_lables
test_correct_preds = (test_preds > MIN_ERROR).astype(int)
test_labels = (test_lables > MIN_ERROR).astype(int)
test_f1 = f1_score(test_labels, test_correct_preds, average="macro", zero_division=0)

accuracy = {}

for og, pred in zip(xd, test_preds):
    for i, (o, p) in enumerate(zip(og, pred)):
        if abs(o - p) < 0.05:
            accuracy[i] = accuracy.get(i, 0) + 1

accuracy = {k: v / len(test_preds) for k, v in accuracy.items()}

overall_accuracy = sum(accuracy.values()) / len(accuracy)


print(f"Macro F1: {test_f1}")
print("Threshold accuracy per class:")
for i, acc in accuracy.items():
    print(f"{id2class[i]}: \t{acc}")
print(f"Overall accuracy: {overall_accuracy}")
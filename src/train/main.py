import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from ..preprocess.sets_preparations import load_sets
import os

if not os.path.exists("logs"):
    os.makedirs("logs")

if not os.path.exists("models"):
    os.makedirs("models")

writer = SummaryWriter("logs/eng_large")

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 1e-5

writer.add_hparams({'MAX_LEN': MAX_LEN, 'BATCH_SIZE': BATCH_SIZE, 'EPOCHS': EPOCHS, 'LEARNING_RATE': LEARNING_RATE},{})

train_df, val_df, test_df = load_sets("data/splits/")

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

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ToxicCommentsDataset(train_df, tokenizer, MAX_LEN)
val_dataset = ToxicCommentsDataset(val_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=7,
    problem_type="multi_label_classification",
)

# PATH = 'models/large_pl_24.pth'
# model.load_state_dict(torch.load(PATH))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    all_labels = []
    all_preds = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.sigmoid(logits).cpu().detach().numpy())

    return np.mean(losses), np.array(all_labels), np.array(all_preds)

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

MIN_ERROR = 0.05

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    train_loss, train_labels, train_preds = train_epoch(model, train_loader, optimizer, device)
    val_labels, val_preds = eval_model(model, val_loader, device)

    train_correct_preds = (train_preds > MIN_ERROR).astype(int)
    val_correct_preds = (val_preds > MIN_ERROR).astype(int)

    train_labels = (train_labels > MIN_ERROR).astype(int)
    val_labels = (val_labels > MIN_ERROR).astype(int)
    train_f1 = f1_score(train_labels, train_correct_preds, average="macro", zero_division=0)
    val_f1 = f1_score(val_labels, val_correct_preds, average="macro", zero_division=0)

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Macro F1/Train", train_f1, epoch)
    writer.add_scalar("Macro F1/Val", val_f1, epoch)
    print(f'Loss: {train_loss} F1 Macro Train: {train_f1} F1 Macro Val: {val_f1}')


torch.save(model.state_dict(), f"models/model_{epoch}.pth")


# Test the model
test_dataset = ToxicCommentsDataset(test_df, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

test_lables, test_preds = eval_model(model, test_loader, device)
test_correct_preds = (test_preds > MIN_ERROR).astype(int)
test_labels = (test_lables > MIN_ERROR).astype(int)
test_f1 = f1_score(test_labels, test_correct_preds, average="macro", zero_division=0)

print(f'Macro F1 test: {test_f1}')
writer.add_scalar("Macro F1/Test", test_f1, 0)

writer.close()
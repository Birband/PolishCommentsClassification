from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained model and tokenizer
model_name = "models/model_0.pth"
device = "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=7)

# Load the model weights
model.load_state_dict(torch.load(model_name, device), device)
text = "Tutaj brzydki komentarz"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", device=device)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted probabilities
probs = torch.sigmoid(outputs.logits)

# Print predicted probabilities
print(probs)
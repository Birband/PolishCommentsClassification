import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "models/large_pl_28.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=7)

model.load_state_dict(torch.load(model_name, device), device)

st.title('Toxicity Classifier')
st.write('Wprowadź tekst, a aplikacja zwróci predykcję toksyczności.')

text = st.text_area("Wprowadź tekst:")

if text:
    inputs = tokenizer(text, return_tensors="pt", device=device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)

    st.write("Predykcja toksyczności:")
    st.write(f"Toxicity: {probs[0][0].item():.4f}")
    st.write(f"Severe Toxicity: {probs[0][1].item():.4f}")
    st.write(f"Obscene: {probs[0][2].item():.4f}")
    st.write(f"Threat: {probs[0][3].item():.4f}")
    st.write(f"Insult: {probs[0][4].item():.4f}")
    st.write(f"Identity Attack: {probs[0][5].item():.4f}")
    st.write(f"Sexual Explicit: {probs[0][6].item():.4f}")

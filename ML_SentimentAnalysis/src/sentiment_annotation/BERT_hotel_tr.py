from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "anilguven/bert_tr_turkish_hotel_reviews"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(dim=-1).item()

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[predicted_class_id]

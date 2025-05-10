import pandas as pd
from transformers import pipeline
import os
import ast

# Load data with sentences and detected aspects
df = pd.read_csv("../dataset/aspect_detected.csv")
df["Aspects"] = df["Aspects"].apply(ast.literal_eval)

# Load sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Map 1–5 stars to sentiment label
def star_to_sentiment(star_text):
    if star_text.startswith("1") or star_text.startswith("2"):
        return "Negative"
    elif star_text.startswith("3"):
        return "Neutral"
    else:
        return "Positive"

# Apply sentiment analysis
def analyze_sentiment(text):
    try:
        prediction = sentiment_model(text)[0]
        label = prediction["label"]
        score = prediction["score"]
        sentiment = star_to_sentiment(label)
        return sentiment, score
    except Exception as e:
        print(f"Error analyzing: {text[:50]}... | {e}")
        return "Unknown", 0

df["Sentiment_Label"], df["Sentiment_Score"] = zip(*df["Sentence"].map(analyze_sentiment))

# Save result
os.makedirs("output", exist_ok=True)
df.to_csv("../dataset/aspect_sentiment.csv", index=False)
print("Sentiment analysis complete. Output saved to output/aspect_sentiment.csv")
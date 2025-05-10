import pandas as pd
import ast
import os

# Load sentence-split data
df = pd.read_csv("../dataset/sentences_split.csv")

# Ensure 'Sentences' column is treated as list
df["Sentences"] = df["Sentences"].apply(ast.literal_eval)

# Explode each sentence into a row
df_sentences = df.explode("Sentences").rename(columns={"Sentences": "Sentence"})

# Define Turkish normalization
def normalize_turkish(text):
    replacements = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u'
    }
    return ''.join(replacements.get(c, c) for c in text.lower())

# Define aspect keywords
aspect_keywords = {
    "Food": ["yemek", "kahvaltı", "restoran", "lezzet", "tatlı", "içecek", "büfe"],
    "Staff": ["personel", "hizmet", "güler", "ilgili", "güleryüz", "çalışan", "animasyon"],
    "Room": ["oda", "yatak", "banyo", "mobilya", "klima", "minibar"],
    "Cleanliness": ["temizlik", "hijyen", "düzen", "kirli", "temiz"],
    "Location": ["konum", "merkez", "şehir", "sahil", "ulaşım"],
    "Entertainment": ["animasyon", "etkinlik", "aktivite", "gösteri", "havuz", "oyun", "müzik"],
    "Family": ["çocuk", "aile", "bebek", "çocuklar"],
    "Noise": ["gürültü", "sessiz", "rahatsız", "kalabalık"],
}

# Normalize keywords
normalized_keywords = {
    aspect: [normalize_turkish(k) for k in keywords]
    for aspect, keywords in aspect_keywords.items()
}

# Aspect detection function
def detect_aspects(sentence):
    aspects = []
    normalized_sentence = normalize_turkish(sentence)
    for aspect, keywords in normalized_keywords.items():
        if any(k in normalized_sentence for k in keywords):
            aspects.append(aspect)
    return aspects

# Detect aspects per sentence
df_sentences["Aspects"] = df_sentences["Sentence"].apply(detect_aspects)

# Filter out sentences with no aspects
df_sentences = df_sentences[df_sentences["Aspects"].map(len) > 0]

# Save to output
os.makedirs("output", exist_ok=True)
df_sentences.to_csv("../dataset/aspect_detected.csv", index=False)
print("Aspect detection complete. Output saved to output/aspect_detected.csv")
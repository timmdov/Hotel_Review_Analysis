import pandas as pd
import stanza
from src.utils.config.paths import STEP3_CLEANED

# Load Turkish NLP model
nlp = stanza.Pipeline("tr", processors="tokenize", use_gpu=False)

def split_sentences(text):
    try:
        doc = nlp(text)
        return [sentence.text for sentence in doc.sentences]
    except Exception as e:
        print(f"Error processing: {text[:50]}... | {e}")
        return []

def main():
    df = pd.read_csv(STEP3_CLEANED).head(50) #TODO: Remove head(50) after testing
    df["Sentences"] = df["Review"].astype(str).apply(split_sentences)
    df.to_csv("../dataset/sentences_split.csv", index=False)
    print("Sentence splitting complete.")

if __name__ == "__main__":
    main()
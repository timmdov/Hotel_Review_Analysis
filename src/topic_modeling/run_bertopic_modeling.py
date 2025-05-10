import os

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from src.utils.config.paths import TOPIC_MODELING_DIR, STEP8_MODEL_READY, STEP5_LEMMATIZED, STEP6_NO_STOPWORDS
from src.utils.io.logger import get_logger

logger = get_logger(__name__)

# Path for output
OUTPUT_CSV = os.path.join(TOPIC_MODELING_DIR, "bertopic_results.csv")
TOPIC_WORDS_CSV = os.path.join(TOPIC_MODELING_DIR, "topic_top_words.csv")

# Optional: define manually labeled topic names (if known)
TOPIC_LABELS = {
    0: "Entertainment & Animation",
    1: "Staff & Service",
    2: "Room Issues",
    3: "Personal Thanks",
    4: "Nationality & Language",
    5: "General Experience",
    6: "Negative Experience",
    7: "Beach & Water",
    8: "Pool & Cleanliness",
    9: "Cleanliness & Food",
    10: "Hygiene & Satisfaction",
    11: "Beach Resort",
    12: "Mixed Opinions",
    13: "Golf & Luxury",
    14: "Holiday Summary",
    15: "Staff & Food",
    16: "Nationality Feedback",
    17: "Very Positive",
    18: "Room & Family",
    19: "Conference",
    20: "Bar & Staff",
    21: "Mediocre Feedback",
    22: "Aquapark & Children",
    23: "Facilities Summary",
    24: "Very Negative",
    25: "Pandemic Period",
    26: "Clean but Lacking",
    27: "Reception & Staff",
    28: "Weak Entertainment",
    29: "Strong Recommendation",
    30: "Pandemic & Thanks",
    31: "Kids & Families",
    32: "Complaints",
    33: "Bar Service",
    34: "City Feedback",
    35: "Hotel Praise",
    36: "Mixed Review",
    37: "Hotel Details",
    38: "Spa & Cleanliness",
    39: "Room Complaints",
    40: "Overall Quality"
}


def get_docs(df: pd.DataFrame, column: str = "Review_Clean") -> list[str]:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in input DataFrame.")
    return df[column].dropna().astype(str).tolist()


def train_topic_model(docs: list[str]) -> BERTopic:
    logger.info("Initializing SentenceTransformer embeddings...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    logger.info("Training BERTopic model...")
    model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        calculate_probabilities=True,
        verbose=True
    )
    model.fit(docs)
    return model


def save_topic_keywords(model: BERTopic, save_path: str, top_n_words: int = 10):
    logger.info(f"Saving top {top_n_words} words per topic...")
    topics_data = []

    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        keywords = model.get_topic(topic_id)[:top_n_words]
        word_list = [word for word, _ in keywords]
        topics_data.append({"Topic_ID": topic_id, "Top_Words": ", ".join(word_list)})

    pd.DataFrame(topics_data).to_csv(save_path, index=False)
    logger.info(f"Saved topic keywords to {save_path}")


def run_bertopic_modeling(input_path: str, output_path: str):
    try:
        logger.info(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        docs = get_docs(df)

        topic_model = train_topic_model(docs)
        topics, _ = topic_model.transform(docs)

        logger.info(f"Discovered {len(set(topics)) - (1 if -1 in topics else 0)} topics.")
        df["Topic"] = topics
        df["Topic_Label"] = df["Topic"].map(TOPIC_LABELS).fillna("Uncategorized")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved topic modeling results to: {output_path}")

        save_topic_keywords(topic_model, TOPIC_WORDS_CSV)

    except Exception as e:
        logger.error(f"BERTopic modeling failed: {e}")


if __name__ == "__main__":
    run_bertopic_modeling(STEP8_MODEL_READY, OUTPUT_CSV)
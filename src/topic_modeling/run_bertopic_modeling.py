import os

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP

from src.utils.config.paths import TOPIC_MODELING_DIR, STEP8_MODEL_READY, STEP5_LEMMATIZED, STEP6_NO_STOPWORDS
from src.utils.io.logger import get_logger

logger = get_logger(__name__)

# Path for output
OUTPUT_CSV = os.path.join(TOPIC_MODELING_DIR, "bertopic_results.csv")
TOPIC_WORDS_CSV = os.path.join(TOPIC_MODELING_DIR, "topic_top_words.csv")

TOPIC_LABELS = {
    0: "Room & Food General Experience",
    1: "Bar & Lounge Staff Experience",
    2: "Entertainment Team & Activities",
    3: "Staff & Guest Relations",
    4: "Hotel Experience & Guest Satisfaction",
    5: "Cleanliness & Comfort",
    6: "Nationality & Cultural Impressions",
    7: "Staff Recognition & Management",
    8: "Activities & Clean Environment",
    9: "Music & Entertainment Shows",
    10: "Golf & Venue Features",
    11: "Staff Service & Dining Experience",
    12: "Beach & View Satisfaction",
    13: "Pandemic Measures & Hygiene",
    14: "Evening Shows & Hotel Events",
    15: "Entertainment Staff Praise",
    16: "Hotel & Staff General Feedback",
    17: "Bar & Service Quality",
    18: "Physical Activities & Events",
    19: "Turkish Identity & Dining",
    20: "Child-Friendliness & Family Facilities",
    21: "Kids’ Activities & Facilities",
    22: "Alcohol & Show Reviews",
    23: "Entertainment Team Vibes",
    24: "Food & Show Appreciation",
    25: "Children’s Happiness & Parental Feedback",
    26: "Seasonal Stay Experience",
    27: "Comfort & Wellness (Massage, View)",
    28: "Service Quality & Guest Complaints",
    29: "Praise for Specific Staff",
    30: "Staff Recognition & Atmosphere",
    31: "Wait Times & Hospitality",
    32: "Dining & Location Service",
    33: "Entertainment & Guest Satisfaction",
    34: "Architecture & Atmosphere",
    35: "Cleanliness & Off-Season Dining",
    36: "Food Quality & Child Satisfaction",
    37: "Seasonal Entertainment (Winter/Summer)",
    38: "Accessibility & Staff Praise",
    39: "Dining Elegance & Staff Recognition",
    40: "Front Desk & Drink Service",
    41: "Emergency Response & Efficiency"
}


def get_docs(df: pd.DataFrame, column: str = "Review_Clean") -> list[str]:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in input DataFrame.")
    return df[column].dropna().astype(str).tolist()


def train_topic_model(docs: list[str]) -> BERTopic:
    logger.info("Initializing SentenceTransformer embeddings...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    logger.info("Training BERTopic model...")

    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
    model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        calculate_probabilities=True,
        verbose=True,
        umap_model=umap_model
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
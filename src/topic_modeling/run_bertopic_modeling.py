import os

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP

from src.utils.config.paths import TOPIC_MODELING_DIR, STEP8_MODEL_READY
from src.utils.io.logger import get_logger

logger = get_logger(__name__)

# Path for output
OUTPUT_CSV = os.path.join(TOPIC_MODELING_DIR, "bertopic_results.csv")
TOPIC_WORDS_CSV = os.path.join(TOPIC_MODELING_DIR, "topic_top_words.csv")

TOPIC_LABELS = {
    0: "Entertainment & Animation",
    1: "Room Issues & Cleanliness",
    2: "Staff & Service Quality",
    3: "Beach & Location",
    4: "Bar & Beverage Service",
    5: "General Hotel Experience",
    6: "Nationality & Diversity",
    7: "Cleanliness & Comfort",
    8: "Staff Compliments (Named)",
    9: "Golf & Events",
    10: "Aquapark & Kids Activities",
    11: "Concerts & Nightlife",
    12: "Service & Food Quality",
    13: "Children & Family Service",
    14: "Animation & Named Staff",
    15: "Beach & Sea View",
    16: "Pandemic Experience & Safety",
    17: "Hotel Experience & Staff",
    18: "Staff Support & Appreciation",
    19: "Sports & Entertainment",
    20: "Room Preference & Comfort",
    21: "Cultural Performance",
    22: "International Shows",
    23: "Facilities & Children Area",
    24: "Aquapark Infrastructure",
    25: "Comfort & View",
    26: "Food Experience & Venue",
    27: "Dining & Kitchen Staff",
    28: "Guest Relations & Entertainment",
    29: "Personal Service Recognition",
    30: "Spa & Sports Facilities",
    31: "Service & Hotel Environment",
    32: "Architecture & Ambience",
    33: "Music & Disco Events",
    34: "Room Service & Satisfaction",
    35: "Seasonal Impressions",
    36: "Bar Issues & Service Gaps"
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
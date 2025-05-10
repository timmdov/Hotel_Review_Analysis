import warnings

import pandas as pd
from bertopic import BERTopic

from src.utils.config import STEP3_CLEANED

warnings.filterwarnings("ignore", category=DeprecationWarning)

merged_topic_labels = {
    1: "Entertainment & Animation",
    3: "Entertainment & Animation",
    5: "Entertainment & Animation",
    30: "Entertainment & Animation",

    2: "Bar & Luxury Experience",
    7: "Bar & Luxury Experience",

    6: "Cleanliness & Food Quality",
    9: "Cleanliness & Food Quality",
    26: "Cleanliness & Food Quality",

    10: "Staff & Service",
    12: "Staff & Service",
    25: "Staff & Service",

    4: "Guest Appreciation & Events",
    15: "Guest Appreciation & Events",
    16: "Guest Appreciation & Events",
    17: "Guest Appreciation & Events",

    8: "Family & Kids",
    24: "Family & Kids",
    28: "Family & Kids",
    29: "Family & Kids",

    13: "Nationality & Cultural Impressions",
    19: "Nationality & Cultural Impressions",
    22: "Nationality & Cultural Impressions",

    11: "COVID-19 & Safety",

    14: "Holiday & Seasonal Comments",
    23: "Holiday & Seasonal Comments",

    20: "Conferences & Meetings",

    27: "Nightlife & Disco",

    0: "General Hotel Experience",
    18: "General Hotel Experience"
}


def run_bertopic_modeling(input_csv_path: str, output_csv_path: str, num_topics_to_display: int = 10):
    # Load dataset
    df = pd.read_csv(input_csv_path)
    docs = df["Review"].dropna().astype(str).tolist()

    # Run BERTopic
    print("Training BERTopic model...")
    topic_model = BERTopic(language="multilingual")  # Handles Turkish well
    topics, probs = topic_model.fit_transform(docs)

    # Show top keywords for each topic
    for topic_id in range(len(topic_model.get_topics())):
        words = topic_model.get_topic(topic_id)
        if words:
            print(f"\nTopic {topic_id}: " + ", ".join([word for word, _ in words[:10]]))

    # Show top topic summary
    print("\nTop Topics:")
    print(topic_model.get_topic_info().head(num_topics_to_display))

    # Save per-review topics
    df["Topic"] = topics
    df["Topic_Label"] = df["Topic"].map(merged_topic_labels).fillna("Uncategorized")
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved topic assignments to: {output_csv_path}")

    # Optional: show interactive topic visualization
    print("Opening visualization...")
    topic_model.visualize_topics().show()


if __name__ == "__main__":
    OUTPUT_PATH = "bertopic_results.csv"
    run_bertopic_modeling(STEP3_CLEANED, OUTPUT_PATH)

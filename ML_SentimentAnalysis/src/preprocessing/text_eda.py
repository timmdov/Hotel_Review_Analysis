from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import LEMMATIZED_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def basic_text_eda(input_path: str, text_column: str = "Review_Lemma"):
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        all_words = []
        for text in df[text_column].dropna():
            all_words.extend(text.split())

        counter = Counter(all_words)
        most_common = counter.most_common(30)

        print("\nMost Common Words:")
        for word, count in most_common:
            print(f"{word:<15} {count}")

        df["word_count"] = df[text_column].apply(lambda x: len(str(x).split()))
        plt.hist(df["word_count"], bins=30, color='skyblue', edgecolor='black')
        plt.title("Distribution of Review Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("../dataset/review_wordcount_distribution.png")
        plt.show()

        logger.info("EDA complete. Histogram saved and printed.")

    except Exception as e:
        logger.error(f"Error in EDA: {e}")


if __name__ == "__main__":
    basic_text_eda(
        input_path=LEMMATIZED_PATH,
        text_column="Review_Lemma"
    )

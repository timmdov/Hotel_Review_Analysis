import warnings

import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
from gensim import corpora, models

from src.utils.config import STEP3_CLEANED

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_lda_topic_modeling(num_topics=5):
    # Load dataset
    df = pd.read_csv(STEP3_CLEANED)
    reviews = df["Review_Lemma"].dropna().astype(str).apply(lambda x: x.split()).tolist()

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(reviews)
    corpus = [dictionary.doc2bow(text) for text in reviews]

    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        random_state=42
    )

    # Print the discovered topics
    for idx, topic in lda_model.print_topics(-1):
        print(f"\nTopic {idx}: {topic}")

    # Optional: Visualize
    print("\nGenerating visualization...")
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.display(vis)
    pyLDAvis.save_html(vis, "topic_modeling/lda_visualization.html")


if __name__ == "__main__":
    run_lda_topic_modeling(num_topics=5)

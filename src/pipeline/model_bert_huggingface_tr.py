"""
Pipeline:
  • load CSV
  • predict sentiment via BERT-tr
  • save predictions
  • if ground-truth 'Sentiment' exists → evaluate & write metrics
"""

import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.sentiment_annotation.BERT_hotel_tr import predict_sentiment
from src.utils.config import STEP8_MODEL_READY

# ──────────────────────────────────────────────────────────────
# Name of the column that **contains the review text**
TEXT_COL = "Review_Lemma"    # ← change if your CSV uses a different name
# ──────────────────────────────────────────────────────────────


def run_bert_tr_pipeline(input_path: str, output_path: str) -> None:
    print(f"✅ Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    if TEXT_COL not in df.columns:
        raise ValueError(f"❌ Missing required column: '{TEXT_COL}'")

    df = df.dropna(subset=[TEXT_COL])
    print(f"🔍 Annotating {len(df):,} reviews ...")

    df["Predicted_Sentiment"] = df[TEXT_COL].apply(predict_sentiment)

    # save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to: {output_path}")

    # optional evaluation
    if "Sentiment" in df.columns:
        print("\n📊 Evaluating predictions ...")

        label_map = {
            "0": "negative", "1": "neutral", "2": "positive",
             0 : "negative",  1 : "neutral",  2 : "positive",
        }
        df["Sentiment"] = df["Sentiment"].map(label_map).fillna(df["Sentiment"])

        df["Sentiment"]           = df["Sentiment"].astype(str).str.lower().str.strip()
        df["Predicted_Sentiment"] = df["Predicted_Sentiment"].str.lower().str.strip()

        y_true, y_pred = df["Sentiment"], df["Predicted_Sentiment"]

        acc    = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)
        cm     = confusion_matrix(y_true, y_pred)

        print(f"\n✅ Accuracy: {acc:.4f}")
        print("\n📄 Classification report:\n", report)
        print("\n🧮 Confusion matrix:\n", cm)

        metrics_path = os.path.join(
            os.path.dirname(output_path), "bert_tr_metrics.txt"
        )
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

        print(f"\n📝 Metrics written to: {metrics_path}")
    else:
        print("ℹ️ No 'Sentiment' column found — skipping evaluation.")


# ── CLI entry ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_file = STEP8_MODEL_READY
    output_file = (
        "/Users/teymurmammadov/PycharmProjects/CS401/CS401_ML_Sentiment-Analysis/"
        "ML_SentimentAnalysis/data/results/bert_tr_predictions.csv"
    )
    run_bert_tr_pipeline(input_file, output_file)

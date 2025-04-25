import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.sentiment_annotation.BERT_hotel_tr import predict_sentiment


def run_bert_tr_pipeline(input_path: str, output_path: str):
    print(f"✅ Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    if "Review" not in df.columns:
        raise ValueError("❌ Missing required column: 'Review'")

    df = df.dropna(subset=["Review"])
    print(f"🔍 Annotating {len(df)} reviews...")

    # 🔁 Use original review text for prediction (not preprocessed)
    df["Predicted_Sentiment"] = df["Review"].apply(predict_sentiment)

    # 💾 Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved predictions to: {output_path}")

    # 📊 Evaluate if true labels exist
    if "Sentiment" in df.columns:
        print("\n📊 Evaluating model predictions...")

        # 🔁 Map numeric labels to string classes
        label_map = {"0": "negative", "1": "neutral", "2": "positive", 0: "negative", 1: "neutral", 2: "positive"}
        df["Sentiment"] = df["Sentiment"].map(label_map).fillna(df["Sentiment"])

        # 🔤 Normalize labels for safety
        df["Sentiment"] = df["Sentiment"].astype(str).str.lower().str.strip()
        df["Predicted_Sentiment"] = df["Predicted_Sentiment"].astype(str).str.lower().str.strip()

        y_true = df["Sentiment"]
        y_pred = df["Predicted_Sentiment"]

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n✅ Accuracy: {acc:.4f}")
        print("\n📊 Classification Report:\n", report)
        print("\n🧮 Confusion Matrix:\n", cm)

        # 💾 Save metrics
        metrics_path = os.path.join(os.path.dirname(output_path), "bert_tr_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

        print(f"\n📝 Evaluation results saved to: {metrics_path}")
    else:
        print("ℹ️ No 'Sentiment' column found — skipping evaluation.")


if __name__ == "__main__":
    input_file = "/Users/teymurmammadov/PycharmProjects/CS401/CS401_ML_Sentiment-Analysis/ML_SentimentAnalysis/data/processed/tripadvisor_sentiment_ready.csv"
    output_file = "/Users/teymurmammadov/PycharmProjects/CS401/CS401_ML_Sentiment-Analysis/ML_SentimentAnalysis/data/results/bert_tr_predictions.csv"
    run_bert_tr_pipeline(input_file, output_file)

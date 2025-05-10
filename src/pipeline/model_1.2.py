import os
import pandas as pd
from src.text_vectorization.text_vectorization_sklearn_tr import vectorize_reviews
from src.models.Logistics_Regression_Model_tr import train_logistic_regression
from src.utils.config.paths import STEP8_MODEL_READY


def run_model_two(data_path: str, results_path: str):
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset: {len(df)} rows from {data_path}")

    required_cols = ["Review_Lemma", "Sentiment"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    X, vectorizer = vectorize_reviews(df, text_column="Review_Lemma")
    y = df["Sentiment"].astype(int)

    model, metrics = train_logistic_regression(X, y)

    # Ensure results dir exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("📊 Model Evaluation Metrics\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-score:  {metrics['f1']:.4f}\n")
        f.write("Confusion Matrix:\n")
        for row in metrics['confusion_matrix']:
            f.write(" ".join(str(x) for x in row) + "\n")

    print(f"📁 Results saved to: {results_path}")
    print("✅ Model 1.2 pipeline complete.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = STEP8_MODEL_READY
    results_file = os.path.join(current_dir, "..", "results", "model_two_evaluation.txt")

    run_model_two(data_path, results_file)

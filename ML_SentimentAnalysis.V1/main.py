import os
import pandas as pd
from text_vectorization.text_vectorization_sklearn import vectorize_reviews
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        multi_class='auto'
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
    cm = confusion_matrix(y_test, predictions)

    print("Evaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return model


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "Booking.com", "hotels_annotated.csv")
    df = pd.read_csv(data_path)
    X, vectorizer = vectorize_reviews(df)
    y = df["Negative_Review_Sentiment"]
    model = train_logistic_regression(X, y)


if __name__ == "__main__":
    main()

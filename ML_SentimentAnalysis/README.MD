# 🏨 Sentiment Analysis of Hotel Reviews — Model One

## 📖 Project Overview

This project is part of a broader effort to build an affordable and accessible sentiment analysis solution for the hotel industry. The goal is to analyze large-scale hotel reviews to help hospitality businesses better understand customer experiences, emotional tones, and service-related patterns. Unlike commercial solutions like ReviewPro or TrustYou, this framework is intended to be open, adaptable, and detailed — focusing not just on positive/negative ratings but on the deeper **emotional and cognitive sentiment** within reviews.

Key project goals:
- 📊 Create a usable dataset of hotel reviews.
- 🧠 Apply **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to classify sentiment.
- 🧩 Cluster reviews and extract aspect-level insights.
- 💬 Empower hotels with actionable feedback from customer voices.

---

## 🧪 Model One — Logistic Regression + VADER

**Model One** is the baseline sentiment analysis pipeline built using traditional machine learning techniques. It uses **logistic regression** to classify review sentiment into **positive**, **negative**, or **neutral** categories, based on VADER-generated labels. This model serves as the foundation for evaluating more advanced methods in future models (e.g., BERT or LSTM-based classifiers).

---

## ⚙️ Technical Specification

### 1. 💾 Dataset
- Source: Scraped or publicly available hotel reviews (e.g., TripAdvisor, Booking.com)
- Language: English (filtered using `langdetect`)
- Labels: Automatically annotated using VADER sentiment scores

### 2. 🧼 Preprocessing
- Convert text to lowercase
- Remove punctuation, emojis, and special characters
- Tokenize into words
- Remove stopwords and short tokens (<3 characters)
- Lemmatize words using spaCy or NLTK

### 3. 🏷 Sentiment Annotation
Using **VADER**:
- Compound score ≥ 0.05 → **Positive**
- Compound score ≤ -0.05 → **Negative**
- Else → **Neutral**

### 4. 🔡 Text Vectorization
- **CountVectorizer** from `scikit-learn`
- Unigrams and bigrams (1,2)
- Max features: 10,000
- Remove terms with document frequency <2

### 5. 🧠 Model
- **Logistic Regression**
  - `solver='lbfgs'`
  - `max_iter=1000`
  - `multi_class='auto'`
- Trained on 80% of the data, validated on 20%

### 6. 📏 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score (macro-averaged)
- Confusion Matrix

---

## 🚧 Next Steps
- Build **Model Two** using BERT or LSTM
- Implement **Aspect-Based Sentiment Analysis (ABSA)**
- Add clustering for review grouping
- Develop a simple dashboard for visualizing insights

---

## 🧠 Authors & Acknowledgment
This project was developed as part of the **CS401 Machine Learning and NLP Project** at the request of the department to explore sentiment-based analytics for the tourism industry.
The authors are: Teymur Mammadov, Rashad Musayev

---

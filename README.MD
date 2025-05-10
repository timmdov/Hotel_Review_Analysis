# 🏨 Sentiment Analysis of Hotel Reviews — Model One

## 📖 Project Overview

This project is part of a broader effort to build an affordable and accessible sentiment analysis solution for the hotel industry. The goal is to analyze large-scale hotel reviews to help hospitality businesses better understand customer experiences, emotional tones, and service-related patterns.

Unlike commercial platforms such as ReviewPro or TrustYou, this framework is intended to be open, adaptable, and detailed — focusing not just on positive/negative ratings but on the deeper **emotional and cognitive sentiment** within reviews.

### 🎯 Project Goals
- 📊 Create a usable dataset of hotel reviews.
- 🧠 Apply **Machine Learning (ML)** and **Natural Language Processing (NLP)** to classify sentiment.
- 🧩 Cluster reviews and extract aspect-level insights.
- 💬 Empower hotels with actionable feedback from customer voices.

---

## 📥 Data Collection

To build our dataset, we used the **TripAdvisor Reviews Scraper** template on [Octoparse](https://www.octoparse.com/) to extract real customer reviews from 10 top-rated hotels in Türkiye.

### ✅ Scraper Configuration
- Start date: `1900-01-01` (to collect **all historical reviews**)
- Max 1,000 reviews per hotel
- Fields collected: `Hotel Name`, `Review`, `Rating`, `Timestamp`

### 🏨 Scraped Hotel URLs

1. [Hotel Sultania — Istanbul](https://www.tripadvisor.com/Hotel_Review-g293974-d1732739-Reviews-Hotel_Sultania-Istanbul.html)  
2. [Divan Cave House — Göreme, Cappadocia](https://www.tripadvisor.com/Hotel_Review-g297983-d1773066-Reviews-Divan_Cave_House-Goreme_Cappadocia.html)  
3. [Liberty Hotels Lara — Antalya](https://www.tripadvisor.com/Hotel_Review-g20116893-d568401-Reviews-Liberty_Hotels_Lara-Kemeragzi_Antalya_Turkish_Mediterranean_Coast.html)  
4. [Lara Barut Collection — Antalya](https://www.tripadvisor.com/Hotel_Review-g15300585-d557049-Reviews-Lara_Barut_Collection-Muratpasa_Antalya_Turkish_Mediterranean_Coast.html)  
5. [Romance Istanbul Hotel](https://www.tripadvisor.com/Hotel_Review-g293974-d8364987-Reviews-Romance_Istanbul_Hotel-Istanbul.html)  
6. [Concorde De Luxe Resort — Antalya](https://www.tripadvisor.com/Hotel_Review-g20116893-d572784-Reviews-Concorde_De_Luxe_Resort-Kemeragzi_Antalya_Turkish_Mediterranean_Coast.html)  
7. [Sheraton Istanbul City Center](https://www.tripadvisor.com/Hotel_Review-g293974-d14930175-Reviews-Sheraton_Istanbul_City_Center-Istanbul.html)  
8. [The Story Hotel Pera — Istanbul](https://www.tripadvisor.com/Hotel_Review-g293974-d2049041-Reviews-The_Story_Hotel_Pera-Istanbul.html)  
9. [Rixos Premium Belek](https://www.tripadvisor.com/Hotel_Review-g312725-d507974-Reviews-Rixos_Premium_Belek_The_Land_Of_Legends_Access-Belek_Serik_District_Turkish_Mediterrane.html)  
10. [Aza Cave Hotel — Göreme, Cappadocia](https://www.tripadvisor.com/Hotel_Review-g297983-d18220399-Reviews-Aza_Cave_Hotel-Goreme_Cappadocia.html)

---

## 🧼 Preprocessing Pipeline

Our preprocessing pipeline is tailored for **Turkish hotel reviews** and ensures clean, consistent input for modeling.

### 🔁 Steps

1. **Filter Columns**
   - Retain only: `HotelName`, `Review`, `Rating`

2. **Language Filtering**
   - Keep only Turkish reviews using `langdetect`

3. **Row Cleaning**
   - Drop duplicates, nulls, and empty reviews

4. **Text Normalization**
   - Lowercase text
   - Remove digits, punctuation, emojis, excessive whitespace

5. **Stopword Removal**
   - Turkish stopwords (`nltk`)
   - Domain-specific terms (e.g., `otel`, `teşekkür`, `bey`, `eder`, `gel`, `hanım`)
   - Remove tokens <3 characters

6. **Lemmatization**
   - Extract root forms using `trnlp` lemmatizer  
   - e.g., *geldim*, *geleceğim* → *gel*

7. **Sentiment Labeling**
   - Based on star rating:
     - `1`–`2` → 0 (Negative)
     - `3`     → 1 (Neutral)
     - `4`–`5` → 2 (Positive)

---

## 📦 Dataset Versions

| Filename                                | Description |
|-----------------------------------------|-------------|
| `tripadvisor_raw_reviews.csv`           | Scraped from Octoparse |
| `tripadvisor_filtered_columns.csv`      | Selected columns only |
| `tripadvisor_tr_reviews.csv`            | Turkish-only reviews |
| `tripadvisor_cleaned_reviews.csv`       | Nulls, duplicates removed |
| `tripadvisor_normalized_reviews.csv`    | Cleaned and lowercased |
| `tripadvisor_cleaned_no_stopwords.csv`  | Stopwords removed |
| `tripadvisor_lemmatized_reviews.csv`    | Lemmatized version |
| `tripadvisor_labeled_reviews.csv`       | Includes `Sentiment` column |
| `tripadvisor_sentiment_ready.csv`       | Final cleaned dataset |
| `tripadvisor_final_model_input.csv`     | Only `Review_Lemma` and `Sentiment` for modeling |

---

## 📊 Exploratory Data Analysis (EDA)

- Performed word frequency analysis after stopword removal  
- Identified **non-informative but frequent** tokens like `teşekkür`, `eder`, `bey`, etc.  
- These were added to the stopword list  
- Plotted distribution of review lengths to validate content richness

---

## 🔡 Text Vectorization

We applied **TF-IDF vectorization** on the `Review_Lemma` column to prepare for model training.

- Library: `TfidfVectorizer` (scikit-learn)
- `ngram_range = (1, 2)` → unigrams and bigrams
- `max_features = 5000`
- `min_df = 5`, `max_df = 0.85`

---

## 🧪 Model One — Logistic Regression

Our first model is a **baseline classifier** using logistic regression trained on the final processed dataset.

- Model: `LogisticRegression`
  - `solver = 'lbfgs'`
  - `max_iter = 1000`
  - `multi_class = 'auto'`
- Train/test split: 80/20
- Evaluation metrics:
  - ✅ Accuracy
  - ✅ Precision, Recall, F1-score (macro avg)
  - ✅ Confusion matrix

---

## 🚧 Next Steps

- ⚙️ Build **Model Two** using BERT or LSTM
- 🧭 Implement **Aspect-Based Sentiment Analysis (ABSA)**
- 📚 Add topic clustering (e.g., LDA)
- 📊 Build a dashboard for result visualization

---

## 🧠 Authors & Acknowledgment

This project was developed as part of the **CS401 Machine Learning and NLP Graduation Project**, supervised by the department.

**Contributors**:
- Teymur Mammadov  
- Rashad Musayev

---
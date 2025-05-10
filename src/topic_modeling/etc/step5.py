import pandas as pd
import os
import matplotlib.pyplot as plt

# Load original review data to restore HotelName info
df_reviews = pd.read_csv("../dataset/sentences_split.csv")  # includes HotelName and Review
df_aspects = pd.read_csv("../dataset/aspect_summary.csv")   # includes Review, Aspect, Sentiment_Label

# Merge to bring HotelName into aspect summary
df = pd.merge(df_aspects, df_reviews[["Review", "HotelName"]], on="Review", how="left")

# Step 1: Count aspect mentions per hotel
mention_counts = df.groupby(["HotelName", "Aspect"]).size().reset_index(name="MentionCount")

# Step 2: Sentiment distribution per aspect per hotel
sentiment_dist = df.groupby(["HotelName", "Aspect", "Sentiment_Label"]).size().reset_index(name="Count")

# Save outputs
os.makedirs("output", exist_ok=True)
mention_counts.to_csv("../dataset/aspect_mention_counts.csv", index=False)
sentiment_dist.to_csv("../dataset/aspect_sentiment_distribution.csv", index=False)
print("Visual data exported.")

# Step 3 (Optional): Example chart per hotel
def plot_aspect_sentiment(hotel_name):
    subset = sentiment_dist[sentiment_dist["HotelName"] == hotel_name]
    pivot = subset.pivot(index="Aspect", columns="Sentiment_Label", values="Count").fillna(0)
    pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title(f"Aspect Sentiment Distribution for {hotel_name}")
    plt.ylabel("Sentence Count")
    plt.xlabel("Aspect")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"output/{hotel_name.replace(' ', '_')}_aspect_sentiment.png")
    plt.close()

# Example: Plot for 1 hotel
plot_aspect_sentiment("Sueno Hotels Deluxe Belek")
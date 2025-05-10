import pandas as pd
import ast
import os
from collections import Counter

# Load sentence-level sentiment results
df = pd.read_csv("../dataset/aspect_sentiment.csv")
df["Aspects"] = df["Aspects"].apply(ast.literal_eval)

# Step 1: Filter low-confidence predictions
df = df[df["Sentiment_Score"] >= 0.5].copy()

# Step 2: Explode aspects so each sentence-aspect pair is a row
df_exploded = df.explode("Aspects").rename(columns={"Aspects": "Aspect"})

# Step 3: Group by original Review (or Review ID if available) and Aspect
# For now, we’ll use full review text as ID — replace if you have a proper ID
if "Review" in df_exploded.columns:
    group_cols = ["Review", "Aspect"]
else:
    group_cols = ["Sentence", "Aspect"]  # fallback

# Step 4: Majority voting per (Review, Aspect)
def majority_vote(sentiments):
    count = Counter(sentiments)
    return count.most_common(1)[0][0]

agg_df = df_exploded.groupby(group_cols).agg({
    "Sentiment_Label": majority_vote,
    "Sentiment_Score": "mean"
}).reset_index()

# Save output
os.makedirs("output", exist_ok=True)
agg_df.to_csv("../dataset/aspect_summary.csv", index=False)
print("Aspect-level sentiment aggregation complete. Saved to output/aspect_summary.csv")
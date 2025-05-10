from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def vader_sentiment_label(text: str) -> str:
    """
    Annotate sentiment using VADER scores:
    - Compound score ≥ 0.05 => positive
    - Compound score ≤ -0.05 => negative
    - Else => neutral
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound_score = scores["compound"]
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

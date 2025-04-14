import pandas as pd
import asyncio
from googletrans import Translator

df = pd.read_csv("../ML_SentimentAnalysis.V1/dataset/TripAdvisor Reviews Scraper.csv")
translator = Translator()

async def translate_text(text):
    try:
        return (await translator.translate(str(text), src='en', dest='tr')).text
    except:
        return ""

async def main():
    tasks = [translate_text(text) for text in df["Review"].astype(str)]
    translated = await asyncio.gather(*tasks)
    df["Review_Text_TR"] = translated
    df.to_csv("../ML_SentimentAnalysis.V1/dataset/Translated Reviews.csv", index=False)

asyncio.run(main())
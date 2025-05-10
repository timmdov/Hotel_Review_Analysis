import asyncio

import pandas as pd
from googletrans import Translator

from src.utils.config import RAW_PATH

df = pd.read_csv(RAW_PATH)
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
    df.to_csv("../data/Translated Reviews.csv", index=False)


asyncio.run(main())

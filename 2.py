from collections import Counter
import pandas as pd
import re

df = pd.read_csv("data/raw/corpus.csv")

words = [
    "cloud", "virus", "stream", "platform",
    "network", "tablet", "model",
    "water", "city", "winter", "animal"
]

tokens = []
for text in df["text"]:
    tokens.extend(re.findall(r"\b[a-z]+\b", str(text).lower()))

freq = Counter(tokens)

for w in words:
    print(w, freq[w])
import pandas as pd

df = pd.read_json("data/raw/News_Category_Dataset_v3.json", lines=True)

df["year"] = pd.to_datetime(df["date"]).dt.year
df["text"] = df["headline"].fillna("") + ". " + df["short_description"].fillna("")

df = df[["year", "text"]]
df = df[df["text"].str.split().str.len() >= 10]

df.to_csv("data/raw/corpus.csv", index=False)

print(df.head())
print(df["year"].value_counts().sort_index())
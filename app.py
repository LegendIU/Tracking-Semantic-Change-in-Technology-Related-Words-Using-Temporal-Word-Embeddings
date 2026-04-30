from pathlib import Path
import pandas as pd
import streamlit as st

RESULTS_DIR = Path("results")

cosine_shift = pd.read_csv(RESULTS_DIR / "cosine_shift.csv")
neighbors = pd.read_csv(RESULTS_DIR / "nearest_neighbors.csv")

st.title("Temporal Semantic Shift Explorer")

word = st.text_input("Enter a word", "virus").lower().strip()

if word:
    word_shift = cosine_shift[cosine_shift["word"] == word]
    word_neighbors = neighbors[neighbors["word"] == word]

    if word_shift.empty:
        st.warning("This word was not found in the current results.")
    else:
        st.subheader("Cosine Shift")
        st.dataframe(word_shift)

        latest = word_shift[word_shift["period"] == "2020s"]
        if not latest.empty:
            score = latest["cosine_distance"].iloc[0]
            st.metric("2020s shift from base period", round(score, 3))

        st.subheader("Nearest Neighbors")
        st.dataframe(word_neighbors)

        if not latest.empty:
            if score > 0.5:
                st.info("Strong semantic shift detected.")
            elif score > 0.3:
                st.info("Moderate semantic shift detected.")
            else:
                st.info("Weak or limited semantic shift detected.")
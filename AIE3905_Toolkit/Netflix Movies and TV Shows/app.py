from pathlib import Path
import sys

import streamlit as st
from study import coverage_summary, recommend, train

st.set_page_config(page_title="Netflix Content Recommendation XAI", layout="wide")

@st.cache_resource
def get_run():
    return train()

run = get_run()
st.title("Netflix Content Recommendation XAI")
st.caption("Explainable content-based recommendation using TF-IDF cosine similarity")
st.markdown("This is a recommendation task, not a label-prediction task. Recommendations are based on overlap in show type, genres, description, country, director, and cast. The explanation terms show the strongest shared TF-IDF features behind each similarity score.")
title = st.selectbox("Choose a title", sorted(run.data["title"].dropna().unique()))
selected = run.data.loc[run.data["title"] == title].iloc[0]
st.subheader("Selected title")
st.write(selected["description"])
st.write({"type": selected["type"], "genres": selected["listed_in"], "country": selected["country"], "release_year": int(selected["release_year"])})
st.subheader("Recommendations with explanations")
st.dataframe(recommend(run, title), hide_index=True, use_container_width=True)
st.caption("Similarity is descriptive. It is not a measure of quality, safety, or user preference, and may amplify missing or uneven metadata.")
st.subheader("Catalogue coverage")
st.dataframe(coverage_summary(run), hide_index=True, use_container_width=True)

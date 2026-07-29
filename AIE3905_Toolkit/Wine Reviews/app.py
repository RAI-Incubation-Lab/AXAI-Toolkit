from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from study import country_audit, global_token_importance, local_token_contributions, train

st.set_page_config(page_title="Wine Review Quality XAI", layout="wide")

@st.cache_resource
def get_run():
    return train()

run = get_run()
st.title("Wine Review Quality XAI")
st.caption("Text classification: predict whether a review score is 90 points or above")
st.markdown("This case uses TF-IDF and logistic regression rather than a tree model. Its explanations are direct word-level contributions: each token's TF-IDF value is multiplied by the fitted linear coefficient.")
cols = st.columns(2)
for col, (name, value) in zip(cols, run.metrics.items()):
    col.metric(name.upper().replace("_", " "), f"{value:.3f}")

index = st.selectbox("Review to inspect", range(len(run.test)), format_func=lambda i: run.test.iloc[i]["title"])
review = run.test.iloc[index]
st.subheader("Selected review")
st.write(review["description"])
st.write({"country": review["country"], "variety": review["variety"], "actual_high_score": int(review["high_score"]), "predicted_high_score": int(run.predictions[index])})

left, right = st.columns(2)
with left:
    st.subheader("Local word contributions")
    st.markdown("Positive values push this review toward the high-score class; negative values push it away. These are associations in review language, not universal quality rules.")
    st.dataframe(local_token_contributions(run, index), hide_index=True, use_container_width=True)
with right:
    st.subheader("Global vocabulary coefficients")
    st.markdown("Large coefficients identify terms the model associates strongly with either class across the corpus.")
    st.dataframe(global_token_importance(run), hide_index=True, use_container_width=True)

st.subheader("Country-level diagnostic")
st.markdown("This compares actual and predicted high-score rates for countries with enough test reviews. It is a sampling diagnostic, not a quality ranking of countries.")
st.dataframe(country_audit(run), hide_index=True, use_container_width=True)

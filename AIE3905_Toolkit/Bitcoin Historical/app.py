from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from study import FEATURES, train

st.set_page_config(page_title="Bitcoin Market Direction XAI", layout="wide")
st.title("Bitcoin Market Direction XAI")
st.caption("Chronological classification: predict whether the next hourly close rises")
st.markdown("This case turns high-frequency price data into lagged return and volume features, then uses a chronological split to avoid training on future periods. It is a teaching exercise, not trading advice.")
try:
    @st.cache_resource
    def get_run():
        return train()
    run = get_run()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

cols = st.columns(2)
for col, (name, value) in zip(cols, run.metrics.items()):
    col.metric(name.upper().replace("_", " "), f"{value:.3f}")
st.subheader("Most recent held-out observations")
st.dataframe(run.test[FEATURES + ["up_next_hour"]].tail(20), hide_index=True, use_container_width=True)
st.subheader("Tree feature importance")
importance = pd.DataFrame({"feature": FEATURES, "importance": run.model.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False)
st.dataframe(importance, hide_index=True, use_container_width=True)
st.info("Market direction is noisy. Accuracy alone is not a trading-performance metric, and explanations of historical features do not imply predictable or causal price movement.")

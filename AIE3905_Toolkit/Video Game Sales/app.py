from pathlib import Path

import pandas as pd

from study import feature_importance, group_audit, lime_local_explanation, shap_summary


def render_tabular_dashboard(study) -> None:
    """Render a consistent, teaching-focused dashboard for a tabular study module."""
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title=study.TITLE, layout="wide")

    @st.cache_resource
    def get_run():
        return study.train()

    run = get_run()
    st.title(study.TITLE)
    st.caption(study.SUBTITLE)
    st.markdown(study.LEARNING_GOAL)
    with st.expander("Learning context", expanded=False):
        st.markdown(study.CONTEXT)

    columns = st.columns(len(run.metrics))
    for column, (name, value) in zip(columns, run.metrics.items()):
        column.metric(name.upper().replace("_", " "), f"{value:.3f}")
    st.caption("Metrics are calculated on a held-out test partition. They summarize prediction quality, not fairness or causality.")

    left, right = st.columns(2)
    with left:
        st.subheader("Data and prediction sample")
        selected = st.selectbox("Test-set passenger / record", range(len(run.X_test)), format_func=lambda i: f"Test record {i + 1}")
        row = run.X_test.iloc[[selected]]
        st.dataframe(row, use_container_width=True, hide_index=True)
        prediction = run.predictions[selected]
        if run.task == "classification":
            st.metric("Predicted class", str(prediction))
            if run.scores is not None:
                st.metric("Positive-class probability", f"{run.scores[selected]:.3f}")
        else:
            st.metric("Predicted target", f"{float(prediction):.3f}")
        st.markdown(study.LOCAL_INTERPRETATION)

    with right:
        st.subheader("Global feature importance")
        st.markdown("This table is the forest's split-based importance. It is a quick diagnostic; SHAP below gives a more principled attribution view.")
        st.dataframe(feature_importance(run), use_container_width=True, hide_index=True)
        st.markdown(study.GLOBAL_INTERPRETATION)

    st.divider()
    explain_col, audit_col = st.columns(2)
    output_dir = Path(study.__file__).resolve().parent / "outputs"
    with explain_col:
        st.subheader("SHAP and LIME")
        st.markdown("SHAP summarizes feature contributions across the model. LIME approximates the model near the selected record with a simple local surrogate.")
        if st.button("Generate SHAP summary", key="shap"):
            try:
                path = shap_summary(run, output_dir / "shap_summary.png")
                st.image(str(path), caption="Tree SHAP global summary")
            except RuntimeError as exc:
                st.warning(str(exc))
        if st.button("Generate LIME explanation", key="lime"):
            try:
                path, weights = lime_local_explanation(run, selected, output_dir / "lime_local.html")
                st.dataframe(pd.DataFrame(weights, columns=["feature rule", "local weight"]), hide_index=True)
                components.html(path.read_text(encoding="utf-8"), height=500, scrolling=True)
            except RuntimeError as exc:
                st.warning(str(exc))

    with audit_col:
        st.subheader(study.AUDIT_TITLE)
        st.markdown(study.AUDIT_EXPLANATION)
        audit = group_audit(run, study.audit_groups(run))
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.info(study.AUDIT_CAUTION)


import study

render_tabular_dashboard(study)

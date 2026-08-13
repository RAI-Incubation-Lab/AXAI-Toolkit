from pathlib import Path

import pandas as pd

from study import feature_importance, group_audit, lime_local_explanation, shap_summary, write_evidence


def render_tabular_dashboard(study) -> None:
    import streamlit as st
    import streamlit.components.v1 as components
    st.set_page_config(page_title=study.TITLE, layout="wide")
    @st.cache_resource
    def get_run(): return study.train()
    run = get_run(); output_dir = Path(study.__file__).resolve().parent / "outputs"; write_evidence(run, output_dir)
    st.title(study.TITLE); st.caption(study.SUBTITLE); st.markdown(study.LEARNING_GOAL)
    with st.expander("Learning context"): st.markdown(study.CONTEXT)
    st.info(f"Validation: {run.split_description}. Model-validity gate: {'passed' if run.validity_passed else 'not passed'}. Do not base decisions on explanations when the gate is not passed.")
    cols = st.columns(len(run.metrics)); [column.metric(name.upper().replace('_', ' '), f"{value:.3f}") for column, (name, value) in zip(cols, run.metrics.items())]
    left, right = st.columns(2)
    with left:
        selected = st.selectbox("Test record", range(len(run.X_test))); row = run.X_test.iloc[[selected]]; st.dataframe(row, hide_index=True, use_container_width=True)
        st.write(f"Ground truth: **{run.y_test.iloc[selected]}**"); st.write(f"Prediction: **{run.predictions[selected]}**")
        if run.scores is not None: st.write(f"Positive-class probability: **{run.scores[selected]:.3f}**")
        st.write(f"Correct: **{bool(run.predictions[selected] == run.y_test.iloc[selected])}**"); st.markdown(study.LOCAL_INTERPRETATION)
    with right: st.dataframe(feature_importance(run), hide_index=True, use_container_width=True); st.markdown(study.GLOBAL_INTERPRETATION)
    explain, audit = st.columns(2)
    with explain:
        if st.button("Generate sampled SHAP summary"):
            try: st.image(str(shap_summary(run, output_dir / "shap_summary.png")), caption="Seeded sample of at most 200 test records")
            except RuntimeError as exc: st.warning(str(exc))
        if st.button("Generate raw-feature LIME explanation"):
            try:
                path, weights, fidelity = lime_local_explanation(run, selected, output_dir / "lime_local.html"); st.caption(f"Local surrogate fidelity: {fidelity:.3f}"); st.dataframe(pd.DataFrame(weights, columns=["feature rule", "local weight"]), hide_index=True); components.html(path.read_text(encoding="utf-8"), height=500, scrolling=True)
            except RuntimeError as exc: st.warning(str(exc))
    with audit:
        st.subheader(study.AUDIT_TITLE); st.markdown(study.AUDIT_EXPLANATION); table = group_audit(run, study.audit_groups(run)); st.dataframe(table, hide_index=True, use_container_width=True); st.warning("This is a subgroup diagnostic, not a formal fairness conclusion. Rows marked small_sample_warning require extra caution."); st.info(study.AUDIT_CAUTION)


import study
render_tabular_dashboard(study)

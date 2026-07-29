from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from titanic_demo import (
    counterfactual_search,
    fairness_audit,
    lime_analysis,
    model_feature_importance,
    passenger_template,
    predict_passenger,
    shap_analysis,
    train_titanic_model,
)


st.set_page_config(page_title="Titanic XAI Demo", page_icon=":ship:", layout="wide")


@st.cache_resource
def load_run():
    return train_titanic_model()


run = load_run()
audit = fairness_audit(run)

st.title("Titanic Survival XAI Demo")
st.caption("RandomForest classifier on Kaggle Titanic passenger data")
st.markdown(
    """
    This dashboard demonstrates a complete explainable AI workflow on the Titanic dataset.
    The model predicts whether a passenger would survive, then exposes the prediction through
    model performance metrics, feature importance, local explanations, fairness checks, and
    counterfactual examples.
    """
)

with st.expander("About this model and dataset", expanded=False):
    st.markdown(
        """
        - **Task:** binary classification, predicting `Survived`.
        - **Model:** scikit-learn `RandomForestClassifier`.
        - **Data source:** Kaggle Titanic dataset with 891 passenger records.
        - **Features used:** ticket class, sex, age, family counts, fare, embarked port,
          engineered `FamilySize`, and `IsAlone`.
        - **Interpretation caution:** this is a historical teaching dataset. Explanations
          reveal model behavior, but they do not prove causal relationships.
        """
    )

metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", f"{run.metrics['accuracy']:.3f}")
metric_cols[1].metric("Precision", f"{run.metrics['precision']:.3f}")
metric_cols[2].metric("Recall", f"{run.metrics['recall']:.3f}")
metric_cols[3].metric("F1", f"{run.metrics['f1']:.3f}")
st.caption(
    "Performance is computed on a stratified 20% test split. Precision and recall are reported "
    "for the positive class: passenger survived."
)

input_col, result_col = st.columns([0.95, 1.05])

with input_col:
    st.subheader("Passenger")
    st.markdown(
        "Adjust the passenger profile below to inspect how the model responds to different "
        "ticket, demographic, family, and fare attributes."
    )
    pclass = st.selectbox("Ticket class", ["1", "2", "3"], index=2)
    sex = st.selectbox("Sex", ["female", "male"], index=1)
    age = st.slider("Age", min_value=0, max_value=80, value=30)
    sibsp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/children aboard", min_value=0, max_value=6, value=0)
    fare = st.slider("Fare", min_value=0.0, max_value=520.0, value=12.0, step=1.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"], index=2)

passenger = passenger_template(
    pclass=pclass,
    sex=sex,
    age=float(age),
    sibsp=int(sibsp),
    parch=int(parch),
    fare=float(fare),
    embarked=embarked,
)
prediction = predict_passenger(run, passenger)

with result_col:
    st.subheader("Prediction")
    st.markdown(
        "The probability bar shows the model's estimated chance of survival. A value at or "
        "above 0.50 is labeled as **Survived**."
    )
    st.metric("Predicted outcome", prediction["label"])
    st.progress(prediction["probability_survived"])
    st.write(f"Survival probability: **{prediction['probability_survived']:.3f}**")

    st.subheader("Counterfactual")
    st.markdown(
        "Counterfactual rows search for nearby passenger profiles that would change the model's "
        "prediction toward survival. They are useful for auditing decision boundaries, but they "
        "should not be treated as real-world causal advice."
    )
    counterfactuals = counterfactual_search(run, passenger)
    st.dataframe(counterfactuals, use_container_width=True, hide_index=True)
    st.caption(
        "Sex is included here as a diagnostic sensitive-attribute test, not as an actionable intervention."
    )

st.divider()

explainer_col, fairness_col = st.columns(2)

with explainer_col:
    st.subheader("Global Explanation")
    st.markdown(
        "Global importance summarizes which encoded features the RandomForest uses most across "
        "the test set. Higher values mean the feature contributes more often to tree decisions."
    )
    st.dataframe(model_feature_importance(run, top_n=12), use_container_width=True, hide_index=True)

    st.markdown(
        "**SHAP** decomposes model output into feature contributions. The summary plot shows "
        "overall influence, while the waterfall plot explains one passenger prediction step by step."
    )
    if st.button("Run SHAP plots"):
        try:
            shap_result = shap_analysis(run)
            st.image(str(shap_result["summary_path"]), caption="SHAP summary plot")
            st.image(str(shap_result["waterfall_path"]), caption="SHAP waterfall for one passenger")
        except RuntimeError as exc:
            st.warning(str(exc))

    st.subheader("Local Explanation")
    st.markdown(
        "**LIME** approximates the model near one passenger and reports feature rules that push "
        "the prediction toward or away from survival. It is local, so its explanation is about "
        "one case rather than the entire model."
    )
    if st.button("Run LIME explanation"):
        try:
            lime_result = lime_analysis(run)
            st.write(pd.DataFrame(lime_result["weights"], columns=["Feature rule", "Weight"]))
            components.html(lime_result["html_path"].read_text(encoding="utf-8"), height=520, scrolling=True)
        except RuntimeError as exc:
            st.warning(str(exc))

with fairness_col:
    st.subheader("Fairness Audit")
    st.markdown(
        "The audit compares prediction behavior across sensitive groups. A larger demographic "
        "parity gap means groups receive positive predictions at very different rates. A larger "
        "equalized odds gap means error behavior differs across groups."
    )
    st.write("Gender")
    st.dataframe(audit["gender_rates"], use_container_width=True, hide_index=True)
    st.write(
        f"Demographic parity gap: **{audit['gender_demographic_parity_gap']:.3f}**  \n"
        f"Equalized odds gap: **{audit['gender_equalized_odds_gap']:.3f}**"
    )

    st.write("Age group")
    st.dataframe(audit["age_rates"], use_container_width=True, hide_index=True)
    st.write(
        f"Demographic parity gap: **{audit['age_demographic_parity_gap']:.3f}**  \n"
        f"Equalized odds gap: **{audit['age_equalized_odds_gap']:.3f}**"
    )
    st.info(
        "In this demo, gender differences are large because the historical labels strongly encode "
        "the evacuation pattern. This makes the dataset useful for teaching fairness diagnostics, "
        "but it also shows why sensitive features require careful justification."
    )

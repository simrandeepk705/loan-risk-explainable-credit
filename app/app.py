# =========================================================
# Loan Approval System with Explainable Credit Risk
# =========================================================
import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json
from datetime import datetime

st.set_page_config(page_title="Loan Risk Decision System", layout="wide")


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    best_xgb = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    shap_explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.joblib"))
    feature_labels = joblib.load(os.path.join(MODELS_DIR, "feature_labels.joblib"))
    thresholds_dict = joblib.load(os.path.join(MODELS_DIR, "decision_thresholds.joblib"))
    feature_names = preprocessor.get_feature_names_out()
    return preprocessor, best_xgb, shap_explainer, feature_labels, thresholds_dict, feature_names

preprocessor, best_xgb, shap_explainer, feature_labels, thresholds_dict, feature_names = load_artifacts()

NUMERIC_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines",
    "NumberOfDependents", "income_missing", "dependents_missing",
    "debt_to_income_ratio", "late_payment_score", "dependents_per_income",
    "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
CATEGORICAL_FEATURES = ["utilization_bucket", "age_bucket"]

def engineer_features(raw_input: dict) -> pd.DataFrame:
    monthly_income = raw_input.get("MonthlyIncome")
    income_missing = int(monthly_income is None or pd.isna(monthly_income))
    monthly_income_filled = monthly_income if not income_missing else 6000

    dependents = raw_input.get("NumberOfDependents")
    dependents_missing = int(dependents is None or pd.isna(dependents))
    dependents_filled = dependents if not dependents_missing else 0

    debt_ratio = raw_input["DebtRatio"]
    debt_to_income_ratio = debt_ratio * monthly_income_filled / (monthly_income_filled + 1)

    late_30_59 = min(raw_input["NumberOfTime30-59DaysPastDueNotWorse"], 18)
    late_60_89 = min(raw_input["NumberOfTime60-89DaysPastDueNotWorse"], 18)
    late_90 = min(raw_input["NumberOfTimes90DaysLate"], 18)
    late_payment_score = late_30_59 * 1 + late_60_89 * 2 + late_90 * 3

    dependents_per_income = dependents_filled / (monthly_income_filled + 1)

    utilization = min(raw_input["RevolvingUtilizationOfUnsecuredLines"], 2.0)
    if utilization <= 0.3:
        utilization_bucket = "low"
    elif utilization <= 0.6:
        utilization_bucket = "medium"
    elif utilization <= 1.0:
        utilization_bucket = "high"
    else:
        utilization_bucket = "over_limit"

    age = raw_input["age"]
    if age <= 30:
        age_bucket = "young"
    elif age <= 45:
        age_bucket = "early_career"
    elif age <= 60:
        age_bucket = "mid_career"
    else:
        age_bucket = "senior"

    row = {
        "RevolvingUtilizationOfUnsecuredLines": utilization,
        "age": age,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income_filled,
        "NumberOfOpenCreditLinesAndLoans": raw_input["NumberOfOpenCreditLinesAndLoans"],
        "NumberRealEstateLoansOrLines": raw_input["NumberRealEstateLoansOrLines"],
        "NumberOfDependents": dependents_filled,
        "income_missing": income_missing,
        "dependents_missing": dependents_missing,
        "debt_to_income_ratio": debt_to_income_ratio,
        "late_payment_score": late_payment_score,
        "dependents_per_income": dependents_per_income,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfTimes90DaysLate": late_90,
        "utilization_bucket": utilization_bucket,
        "age_bucket": age_bucket,
    }
    return pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

def get_reason_codes(shap_row, feature_names_arr, top_n=3):
    contributions = list(zip(feature_names_arr, shap_row))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for feat_name, value in contributions[:top_n]:
        readable = feature_labels.get(feat_name, feat_name.split("__")[-1].replace("_", " "))
        direction = "increased risk" if value > 0 else "decreased risk"
        reasons.append((readable.capitalize(), direction, float(value)))
    return reasons

def decide(score, review_t, reject_t):
    if score >= reject_t:
        return "Reject"
    elif score >= review_t:
        return "Manual Review"
    else:
        return "Approve"

def predict_applicant(raw_input: dict) -> dict:
    X_row = engineer_features(raw_input)
    X_processed = preprocessor.transform(X_row)
    risk_score = float(best_xgb.predict_proba(X_processed)[:, 1][0])
    decision = decide(risk_score, thresholds_dict["approve_below"], thresholds_dict["reject_at_or_above"])
    shap_row = shap_explainer.shap_values(X_processed)[0]
    reasons = get_reason_codes(shap_row, feature_names, top_n=3)
    return {
        "risk_score": risk_score, "decision": decision, "reasons": reasons,
        "shap_row": shap_row, "X_processed": X_processed,
    }


st.sidebar.header("📋 Applicant Information")

utilization = st.sidebar.slider("Credit Utilization Ratio", 0.0, 2.0, 0.45, 0.01,
                                  help="Total balance on credit lines ÷ credit limits")
age = st.sidebar.slider("Age", 18, 90, 34)
debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 3.0, 0.60, 0.01,
                                 help="Monthly debt payments ÷ monthly income")
monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=0, value=4200, step=100)
open_credit_lines = st.sidebar.slider("Open Credit Lines / Loans", 0, 30, 6)
real_estate_loans = st.sidebar.slider("Real Estate Loans", 0, 10, 1)
dependents = st.sidebar.slider("Number of Dependents", 0, 10, 2)

st.sidebar.subheader("Payment History")
late_30_59 = st.sidebar.slider("30-59 Days Late (count)", 0, 15, 1)
late_60_89 = st.sidebar.slider("60-89 Days Late (count)", 0, 15, 0)
late_90 = st.sidebar.slider("90+ Days Late (count)", 0, 15, 0)

raw_input = {
    "RevolvingUtilizationOfUnsecuredLines": utilization,
    "age": age,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": monthly_income,
    "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
    "NumberRealEstateLoansOrLines": real_estate_loans,
    "NumberOfDependents": dependents,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
    "NumberOfTimes90DaysLate": late_90,
}


result = predict_applicant(raw_input)


st.title("🏦 Loan Approval System — Explainable Credit Risk")
st.caption("Adjust the sliders on the left and watch the decision update live.")

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Decision")

    decision_colors = {"Approve": "#16A34A", "Manual Review": "#CA8A04", "Reject": "#DC2626"}
    color = decision_colors[result["decision"]]

    st.markdown(
        f"""
        <div style="background-color:{color}22; border-left: 6px solid {color};
                    padding: 20px; border-radius: 6px;">
            <div style="font-size: 14px; color: #666;">DECISION</div>
            <div style="font-size: 32px; font-weight: 700; color: {color};">{result['decision']}</div>
            <div style="font-size: 14px; color: #666; margin-top: 8px;">RISK SCORE</div>
            <div style="font-size: 24px; font-weight: 600;">{result['risk_score']:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Top Reasons")
    for label, direction, value in result["reasons"]:
        icon = "🔺" if direction == "increased risk" else "🔻"
        st.write(f"{icon} **{label}** {direction}")

    st.markdown("---")
    st.markdown("#### Decision Thresholds")
    st.write(f"Approve: score below **{thresholds_dict['approve_below']:.2f}**")
    st.write(f"Manual Review: **{thresholds_dict['approve_below']:.2f} – {thresholds_dict['reject_at_or_above']:.2f}**")
    st.write(f"Reject: score at or above **{thresholds_dict['reject_at_or_above']:.2f}**")

with col2:
    st.subheader("Why this decision — SHAP Explanation")

    explanation = shap.Explanation(
        values=result["shap_row"],
        base_values=shap_explainer.expected_value,
        data=result["X_processed"][0],
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(explanation, show=False, max_display=10)
    st.pyplot(fig, clear_figure=True)

    st.caption(
        "This chart shows how each of this applicant's features pushed their "
        "risk score up (red) or down (blue) from the average baseline prediction."
    )

st.markdown("---")
st.caption(
    "⚠ This is a demo built on a public dataset for educational purposes. "
)

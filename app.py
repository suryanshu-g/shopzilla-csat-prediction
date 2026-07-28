"""
CSAT Prediction — Local Deployment App
========================================
Run with:  streamlit run app.py

Expects a folder named `preprocessing_objects/` in the same directory,
containing everything saved from the notebook:
    - csat_ann_model.keras
    - scaler.pkl
    - agent_encoder.pkl
    - city_encoder.pkl
    - onehot_column_list.pkl
    - preprocessing_metadata.pkl
    - shap_background.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow import keras

ARTIFACT_DIR = "preprocessing_objects"

# ---------------------------------------------------------------------------
# Load all artifacts once (cached across Streamlit reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(os.path.join(ARTIFACT_DIR, "csat_ann_model.keras"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    agent_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "agent_encoder.pkl"))
    city_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "city_encoder.pkl"))
    onehot_column_list = joblib.load(os.path.join(ARTIFACT_DIR, "onehot_column_list.pkl"))
    metadata = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessing_metadata.pkl"))
    background = joblib.load(os.path.join(ARTIFACT_DIR, "shap_background.pkl"))
    return model, scaler, agent_encoder, city_encoder, onehot_column_list, metadata, background


model, scaler, agent_encoder, city_encoder, onehot_column_list, metadata, background = load_artifacts()

numeric_cols = metadata["numeric_columns"]      # Item_price, response_time_minutes, remark_length, remark_sentiment, hour_of_day
binary_cols = metadata["binary_columns"]        # has_order_info, same_day_response, is_weekend, has_remark
onehot_cols = metadata["onehot_columns"]        # channel_name, category, Sub-category, Product_category, Supervisor, Manager, Tenure Bucket, Agent Shift, day_of_week
agent_unknown_index = metadata["agent_unknown_index"]
city_unknown_index = metadata["city_unknown_index"]

# ---------------------------------------------------------------------------
# Rebuild dropdown option lists directly from the training-time one-hot
# columns, so the app can only offer categories the model actually learned.
# Columns are named "<prefix>_<value>" (e.g. "Sub-category_Missing").
# ---------------------------------------------------------------------------

def options_for(prefix):
    marker = prefix + "_"
    values = [c[len(marker):] for c in onehot_column_list if c.startswith(marker)]
    return sorted(values)

category_options = {col: options_for(col) for col in onehot_cols}

n_dense = len(numeric_cols) + len(binary_cols) + len(onehot_column_list)

# ---------------------------------------------------------------------------
# Preprocessing: turn one raw form submission into the model's 3 real inputs
# (must exactly replicate the notebook's Part 2 pipeline, in the same order:
#  numeric_cols -> binary_cols -> onehot_column_list)
# ---------------------------------------------------------------------------

def preprocess_input(raw):
    # --- numeric branch (scaled with the SAME fitted scaler from training) ---
    numeric_df = pd.DataFrame([[raw[c] for c in numeric_cols]], columns=numeric_cols)
    numeric_scaled = scaler.transform(numeric_df)

    # --- binary branch (already 0/1, no transform needed) ---
    binary_arr = np.array([[raw[c] for c in binary_cols]], dtype="float32")

    # --- one-hot branch: build a single row with all training-time dummy
    #     columns, mark the selected category as 1, everything else 0 ---
    onehot_row = pd.DataFrame(0, index=[0], columns=onehot_column_list, dtype="int8")
    for col in onehot_cols:
        selected_value = raw[col]
        dummy_col = f"{col}_{selected_value}"
        if dummy_col in onehot_row.columns:
            onehot_row.at[0, dummy_col] = 1
        # if the selected value has no matching dummy column, it was the
        # baseline/dropped category (or unseen) — leaving all zeros is correct

    dense_row = np.hstack([numeric_scaled, binary_arr, onehot_row.values]).astype("float32")

    # --- embedding branches, with the same "unknown" fallback used in training ---
    agent_idx = np.array([[agent_encoder.get(raw["Agent_name"], agent_unknown_index)]], dtype="int32")
    city_idx = np.array([[city_encoder.get(raw["Customer_City"], city_unknown_index)]], dtype="int32")

    return dense_row, agent_idx, city_idx


def predict_fn(combined):
    """Wrapper for SHAP: splits one combined 2D array back into the 3 real model inputs."""
    dense_part = combined[:, :n_dense]
    agent_part = combined[:, n_dense:n_dense + 1]
    city_part = combined[:, n_dense + 1:]
    return model.predict([dense_part, agent_part, city_part], verbose=0)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CSAT Predictor", layout="centered")
st.title("Customer Satisfaction (CSAT) Predictor")
st.caption(
    "Predicts a 1-5 CSAT score from support-interaction details. "
    "Trained on Shopzilla support data — macro-F1 ≈ 0.25, best on extreme "
    "ratings (1, 5); CSAT 2-4 are a known hard case (see explanation below)."
)

with st.form("csat_form"):
    st.subheader("Interaction Details")

    col1, col2 = st.columns(2)
    with col1:
        channel_name = st.selectbox("Channel", category_options["channel_name"])
        category = st.selectbox("Category", category_options["category"])
        sub_category = st.selectbox("Sub-category", category_options["Sub-category"])
        product_category = st.selectbox("Product category", category_options["Product_category"])
        item_price = st.number_input("Item price", min_value=0.0, value=500.0, step=50.0)
        has_order_info = st.checkbox("Order details available for this interaction", value=True)

    with col2:
        agent_name = st.selectbox("Agent", sorted(agent_encoder.keys()))
        customer_city = st.selectbox("Customer city", sorted(city_encoder.keys()))
        supervisor = st.selectbox("Supervisor", category_options["Supervisor"])
        manager = st.selectbox("Manager", category_options["Manager"])
        tenure_bucket = st.selectbox("Agent tenure bucket", category_options["Tenure Bucket"])
        agent_shift = st.selectbox("Agent shift", category_options["Agent Shift"])

    st.subheader("Timing")
    col3, col4, col5 = st.columns(3)
    with col3:
        response_time_minutes = st.number_input("Response time (minutes)", min_value=0.0, value=15.0, step=1.0)
    with col4:
        hour_of_day = st.slider("Hour issue reported", 0, 23, 12)
    with col5:
        day_of_week = st.selectbox("Day of week", category_options["day_of_week"])

    same_day_response = st.checkbox("Responded same day", value=True)
    is_weekend = day_of_week in ("Saturday", "Sunday")

    st.subheader("Customer Remark")
    remark_text = st.text_area("Customer remark (optional)", "")

    submitted = st.form_submit_button("Predict CSAT")

if submitted:
    # Derive the remark-based features exactly as done in Part 1 (VADER sentiment)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    has_remark = 1 if remark_text.strip() else 0
    remark_length = len(remark_text)
    remark_sentiment = analyzer.polarity_scores(remark_text)["compound"] if remark_text.strip() else 0.0

    raw = {
        "Item_price": item_price,
        "response_time_minutes": response_time_minutes,
        "remark_length": remark_length,
        "remark_sentiment": remark_sentiment,
        "hour_of_day": hour_of_day,
        "has_order_info": int(has_order_info),
        "same_day_response": int(same_day_response),
        "is_weekend": int(is_weekend),
        "has_remark": has_remark,
        "channel_name": channel_name,
        "category": category,
        "Sub-category": sub_category,
        "Product_category": product_category,
        "Supervisor": supervisor,
        "Manager": manager,
        "Tenure Bucket": tenure_bucket,
        "Agent Shift": agent_shift,
        "day_of_week": day_of_week,
        "Agent_name": agent_name,
        "Customer_City": customer_city,
    }

    dense_row, agent_idx, city_idx = preprocess_input(raw)
    probs = model.predict([dense_row, agent_idx, city_idx], verbose=0)[0]
    predicted_class = int(np.argmax(probs)) + 1  # back to 1-5

    st.subheader("Prediction")
    st.metric("Predicted CSAT Score", f"{predicted_class} / 5")

    prob_df = pd.DataFrame({"CSAT Score": [1, 2, 3, 4, 5], "Probability": probs})
    st.bar_chart(prob_df.set_index("CSAT Score"))

    # --- one-shot SHAP explanation for this specific prediction ---
    with st.spinner("Computing explanation..."):
        import shap
        combined_input = np.hstack([dense_row, agent_idx.astype("float32"), city_idx.astype("float32")])
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(combined_input, nsamples=100, silent=True)

        feature_names = numeric_cols + binary_cols + list(onehot_column_list) + ["Agent_name", "Customer_City"]
        pred_idx = predicted_class - 1
        local_shap = pd.Series(shap_values[0, :, pred_idx], index=feature_names)
        top_features = local_shap.reindex(local_shap.abs().sort_values(ascending=False).index).head(5)

    st.subheader("Why this prediction?")
    for feat, val in top_features.items():
        direction = "pushed the score up" if val > 0 else "pushed the score down"
        st.write(f"- **{feat}** {direction} (impact: {val:+.3f})")

    if predicted_class in (2, 3, 4):
        st.info(
            "Note: this model is known to be less reliable on CSAT 2-4 predictions "
            "(macro-F1 ≈ 0.25 on these classes, vs. much stronger performance on "
            "CSAT 1 and 5). Treat this specific prediction as a rough signal, not "
            "a confident classification — see the project's SHAP analysis for why."
        )
        
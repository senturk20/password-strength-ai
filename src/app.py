import sys
from pathlib import Path

# Add project root to sys.path so `from src.*` imports work when this file
# is executed directly (python src/app.py). Prefer running with
# `streamlit run src/app.py` or `python -m src.app` in normal usage.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import json
from joblib import load
import matplotlib.pyplot as plt

from src.password_features import extract_features
from src.explainer import explain_password
from src.password_generator import suggest_passwords, PasswordPolicy


# ------------------------------------------
# LOAD MODEL + METADATA
# ------------------------------------------
MODEL_PATH = "models/best_model.joblib"
FEATURE_PATH = "models/feature_list.json"
LABEL_MAP_PATH = "models/label_mapping.json"
METRICS_PATH = "models/model_metrics.json"
from pathlib import Path

# Resolve model path with fallback (some scripts save under a different name)
model_path = Path(MODEL_PATH)
if not model_path.exists():
    alt = Path("models/password_strength_model.joblib")
    if alt.exists():
        model_path = alt
    else:
        raise FileNotFoundError(f"No model found at {model_path} or {alt}")

model = load(model_path)
feature_list = json.load(open(FEATURE_PATH))
label_map = json.load(open(LABEL_MAP_PATH))


# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.set_page_config(page_title="Password Strength Analyzer", layout="centered")

st.title("🔐 AI-Powered Password Strength Analyzer")
st.write(
    "This tool evaluates password strength using machine learning and advanced feature engineering.\n"
    "Your password is never stored or transmitted anywhere."
)


# ------------------------------------------
# PASSWORD INPUT
# ------------------------------------------
password = st.text_input("Enter a password:", type="password", help="Your input remains local and private.")

if st.button("Evaluate Password"):
    if len(password) == 0:
        st.warning("Please enter a password.")
    else:
        # Extract features
        feats = extract_features(password)
        vector = np.array([feats[f] for f in feature_list])

        # Predict
        prediction = model.predict([vector])[0]
        probs = model.predict_proba([vector])[0]

        label_text = label_map[str(prediction)]

        # ------------------------------------------
        # SUGGESTED STRONGER PASSWORDS
        # ------------------------------------------
        # prediction == 2 is expected to be 'Strong' in label mapping
        try:
            is_strong = (int(prediction) == 2)
        except Exception:
            is_strong = (label_text.lower() == "strong")

        if not is_strong:
            st.subheader("💡 Suggested Stronger Passwords")
            suggestions = suggest_passwords(password, PasswordPolicy())
            for pw in suggestions:
                st.code(pw, language="text")
        # ------------------------------------------
        # RESULT CARD
        # ------------------------------------------
        st.subheader("🧠 Result")
        st.success(f"**Password Strength:** {label_text}")

        # ------------------------------------------
        # PROBABILITY VISUALIZATION
        # ------------------------------------------
        st.subheader("📊 Strength Probabilities")

        fig, ax = plt.subplots(figsize=(5, 2))
        categories = ["Weak", "Medium", "Strong"]
        ax.bar(categories, probs, color=["#ff4b4b", "#ffa534", "#4CAF50"])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # ------------------------------------------
        # EXPLANATION
        # ------------------------------------------
        st.subheader("🔍 Detailed Explanation")
        explanation = explain_password(feats, prediction)
        st.write(explanation.replace("\n", "  \n"))

        # ------------------------------------------
        # OPTIONAL: Feature Debug Info
        # ------------------------------------------
        with st.expander("🔧 Show extracted features (debug)"):
            st.json(feats)

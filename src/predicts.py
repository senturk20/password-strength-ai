import numpy as np
from joblib import load
from src.password_features import extract_features
from src.explainer import explain_password
from pathlib import Path

# Load model + metadata. Use a sensible fallback if the exact filename
# isn't present (some scripts save as 'password_strength_model.joblib').
model_path = Path("models/best_model.joblib")
if not model_path.exists():
	alt = Path("models/password_strength_model.joblib")
	if alt.exists():
		model_path = alt
	else:
		raise FileNotFoundError(f"No model found at {model_path} or {alt}")

model = load(model_path)

# Label mapping
import json
label_map = json.load(open("models/label_mapping.json", "r"))

password = input("🔐 Enter a password to evaluate: ")

features = extract_features(password)
feature_vector = np.array([features[f] for f in json.load(open("models/feature_list.json"))])

prediction = model.predict([feature_vector])[0]
probs = model.predict_proba([feature_vector])[0]

# Print result
print("\n🧠 Password Strength Prediction")
print("-----------------------------------")
print(f"Prediction: {label_map[str(prediction)]}")
print(f"Probabilities (Weak/Medium/Strong): {probs}")
print("-----------------------------------")

# Explanation
print("\n🔍 Explanation:")
print(explain_password(features, prediction))

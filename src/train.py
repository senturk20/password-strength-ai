import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from joblib import dump
import matplotlib.pyplot as plt

from src.password_features import build_feature_matrix, FEATURE_LIST

# --------------------------------------------
# SETTINGS
# --------------------------------------------
DATA_PATH = Path(r"C:\Users\omers\OneDrive\Masaüstü\password-strength-project\data\dataV2.csv")  # Change if needed
MODEL_PATH = Path("models/password_strength_model.joblib")
LABEL_MAP_PATH = Path("models/label_mapping.json")
FEATURES_PATH = Path("models/feature_list.json")

# --------------------------------------------
# 1) LOAD DATA
# --------------------------------------------
def load_dataset(data_path: Path):
    df = pd.read_csv(data_path, delimiter=";", header=None)
    df.columns = ['password', 'strength', 'new_strength']

    # Keep only passwords + target label
    df = df.dropna(subset=['password', 'new_strength'])
    df['new_strength'] = pd.to_numeric(df['new_strength'], errors='coerce')
    df = df.dropna(subset=['new_strength'])

    return df[['password', 'new_strength']].reset_index(drop=True)

# --------------------------------------------
# 2) LABEL NORMALIZATION
# --------------------------------------------
def normalize_labels(y):
    """
    Normalize strength labels into 3 categories:
        0 = Weak
        1 = Medium
        2 = Strong
    Adjust the mapping here if needed.
    """

    # Example mapping for your 0–8 dataset
    # You can adjust the thresholds later
    def map_strength(val):
        if val <= 2:
            return 0
        elif val <= 5:
            return 1
        else:
            return 2

    y_mapped = y.apply(map_strength)
    label_map = {0: "Weak", 1: "Medium", 2: "Strong"}

    return y_mapped, label_map

# --------------------------------------------
# 3) MAIN TRAINING PIPELINE
# --------------------------------------------
def train_model():

    print("📌 Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("📌 Normalizing labels...")
    y, label_map = normalize_labels(df['new_strength'])

    print("📌 Extracting features...")
    X = build_feature_matrix(df['password'])

    # Train-test split
    print("📌 Creating train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --------------------------------------------
    # 🌳 BASELINE MODEL
    # --------------------------------------------
    print("📌 Training RandomForest baseline model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    # --------------------------------------------
    # Evaluation
    # --------------------------------------------
    print("📌 Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - Password Strength Model")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------
    # SAVE MODEL & METADATA
    # --------------------------------------------
    print("📌 Saving model and metadata...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    dump(model, MODEL_PATH)

    # Save feature ordering
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_LIST, f, indent=4)

    # Save label mapping
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=4)

    print("✅ Training completed successfully!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Feature order saved to: {FEATURES_PATH}")
    print(f"Label mapping saved to: {LABEL_MAP_PATH}")


# --------------------------------------------
# ENTRY POINT
# --------------------------------------------
if __name__ == "__main__":
    train_model()

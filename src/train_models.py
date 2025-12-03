import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when this
# script is executed directly (python src/train_models.py). Prefer running
# with `python -m src.train_models` in production, but this makes the script
# more robust for ad-hoc runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from joblib import dump

import xgboost as xgb
import lightgbm as lgb

from src.password_features import build_feature_matrix, FEATURE_LIST


# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_PATH = Path(r"C:\Users\omers\OneDrive\Masaüstü\password-strength-project\data\dataV2.csv")
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
FEATURE_PATH = MODEL_DIR / "feature_list.json"
LABEL_MAP_PATH = MODEL_DIR / "label_mapping.json"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
def load_dataset(path: Path):
    df = pd.read_csv(path, delimiter=";", header=None)
    df.columns = ["password", "strength", "new_strength"]

    df = df.dropna(subset=["password", "new_strength"])
    df["new_strength"] = pd.to_numeric(df["new_strength"], errors="coerce")
    df = df.dropna(subset=["new_strength"])

    return df.reset_index(drop=True)


# --------------------------------------------------
# LABEL NORMALIZATION
# --------------------------------------------------
def normalize_labels(y):
    def map_strength(val):
        if val <= 2:
            return 0     # weak
        elif val <= 5:
            return 1     # medium
        else:
            return 2     # strong

    y_mapped = y.apply(map_strength)
    label_map = {0: "Weak", 1: "Medium", 2: "Strong"}

    return y_mapped, label_map


# --------------------------------------------------
# TRAIN MODEL + GRID SEARCH
# --------------------------------------------------
def train_with_grid(model, param_grid, X_train, y_train):
    """Perform GridSearchCV with the given model & params."""
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring="f1_macro"
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# --------------------------------------------------
# MODEL TRAINING PIPELINE
# --------------------------------------------------
def train_all_models():

    print("📌 Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("📌 Normalizing labels...")
    y, label_map = normalize_labels(df["new_strength"])

    print("📌 Extracting features...")
    X = build_feature_matrix(df["password"])

    print("📌 Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --------------------------------------------------
    # Model 1: RANDOM FOREST
    # --------------------------------------------------
    print("\n🌲 Training RandomForest (GridSearch Option B)...")

    rf = RandomForestClassifier(random_state=42)

    rf_params = {
        "n_estimators": [200, 400],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    best_rf, best_rf_params = train_with_grid(rf, rf_params, X_train, y_train)

    rf_pred = best_rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")

    print("Best RF Params:", best_rf_params)
    print(classification_report(y_test, rf_pred))


    # --------------------------------------------------
    # Model 2: XGBOOST
    # --------------------------------------------------
    print("\n⚡ Training XGBoost (GridSearch Option B)...")

    xgb_model = xgb.XGBClassifier(
        random_state=42,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss'
    )

    xgb_params = {
        "n_estimators": [200, 400],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    }

    best_xgb, best_xgb_params = train_with_grid(xgb_model, xgb_params, X_train, y_train)

    xgb_pred = best_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average="macro")

    print("Best XGB Params:", best_xgb_params)
    print(classification_report(y_test, xgb_pred))


    # --------------------------------------------------
    # Model 3: LIGHTGBM
    # --------------------------------------------------
    print("\n💡 Training LightGBM (GridSearch Option B)...")

    lgb_model = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=42)

    lgb_params = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [-1, 10, 20],
        "num_leaves": [31, 63]
    }

    best_lgb, best_lgb_params = train_with_grid(lgb_model, lgb_params, X_train, y_train)

    lgb_pred = best_lgb.predict(X_test)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    lgb_f1 = f1_score(y_test, lgb_pred, average="macro")

    print("Best LGB Params:", best_lgb_params)
    print(classification_report(y_test, lgb_pred))


    # --------------------------------------------------
    # COMPARE MODELS
    # --------------------------------------------------
    print("\n📊 Model Comparison (Accuracy & Macro F1):\n")

    results = {
        "RandomForest": {"accuracy": rf_acc, "f1_macro": rf_f1},
        "XGBoost": {"accuracy": xgb_acc, "f1_macro": xgb_f1},
        "LightGBM": {"accuracy": lgb_acc, "f1_macro": lgb_f1},
    }

    print(json.dumps(results, indent=4))


    # --------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = {"RandomForest": best_rf, "XGBoost": best_xgb, "LightGBM": best_lgb}[best_name]

    print(f"\n🏆 BEST MODEL: {best_name}")

    # Save model
    dump(best_model, BEST_MODEL_PATH)

    # Save metadata
    with open(FEATURE_PATH, "w") as f:
        json.dump(FEATURE_LIST, f, indent=4)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=4)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Saved best model: {BEST_MODEL_PATH}")
    print("📁 Model metrics saved!")


    # --------------------------------------------------
    # FEATURE IMPORTANCE PLOTS
    # --------------------------------------------------
    print("\n📌 Plotting Feature Importances...")

    plt.figure(figsize=(10, 6))

    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": FEATURE_LIST,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    plt.barh(feat_df["feature"], feat_df["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    train_all_models()

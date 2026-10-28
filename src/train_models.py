import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. File paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "data" / "splits"
RESULTS_DIR = BASE_DIR / "data" / "model_results"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"

RANDOM_STATE = 42

TEXT_COLUMNS = ["merchant_name", "purchased_items"]


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Please run src/split_dataset.py first."
        )

    return pd.read_csv(path)


# --------------------------------------------------
# 3. Prepare receipt text
# --------------------------------------------------

def prepare_text(df):
    """
    Combine merchant name and purchased items.
    Missing text is replaced with an empty string.
    """
    merchant = df["merchant_name"].fillna("").astype(str)
    items = df["purchased_items"].fillna("").astype(str)

    return (merchant + " " + items).str.strip()


# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

def build_models():
    return {
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "XGBoost": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )),
            ("classifier", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            )),
        ]),
    }


# --------------------------------------------------
# 5. Train and validate
# --------------------------------------------------

def main():
    print("=" * 60)
    print("EXPENSE CLASSIFICATION MODEL TRAINING")
    print("=" * 60)

    train_df = load_dataset(TRAIN_PATH)
    validation_df = load_dataset(VALIDATION_PATH)

    X_train = prepare_text(train_df)
    y_train = train_df["expense_category"].astype(str)

    X_validation = prepare_text(validation_df)
    y_validation = validation_df["expense_category"].astype(str)

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain at least two categories.")

    print("Training records:", len(train_df))
    print("Validation records:", len(validation_df))
    print("\nTraining category counts:")
    print(y_train.value_counts().to_string())

    models = build_models()
    results = []
    trained_models = {}

    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    for name, model in models.items():
        print("\n" + "-" * 60)
        print("Training:", name)

        if name == "XGBoost":
            # XGBoost requires integer labels for this setup.
            y_train_encoded = label_encoder.transform(y_train)

            model.fit(X_train, y_train_encoded)

            encoded_predictions = model.predict(X_validation).astype(int)
            predictions = label_encoder.inverse_transform(
                encoded_predictions
            )
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)

        accuracy = accuracy_score(y_validation, predictions)

        macro_f1 = f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0,
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })

        trained_models[name] = model

        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation macro F1: {macro_f1:.4f}")
        print(f"Validation weighted F1: {weighted_f1:.4f}")

        print("\nValidation classification report:")
        print(classification_report(
            y_validation,
            predictions,
            labels=label_encoder.classes_,
            zero_division=0,
        ))


    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "validation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\nResults saved to:", results_path)

    best_model_name = results_df.iloc[0]["model"]
    print("\nBest model by validation macro F1:", best_model_name)
    print("The test set has not been used.")


if __name__ == "__main__":
    main()

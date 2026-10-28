
import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "data" / "splits"
RESULTS_DIR = BASE_DIR / "data" / "model_results"
MODEL_DIR = BASE_DIR / "models"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

MODEL_PATH = MODEL_DIR / "best_expense_classifier.joblib"

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load dataset
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
    merchant = df["merchant_name"].fillna("").astype(str)
    items = df["purchased_items"].fillna("").astype(str)

    return (merchant + " " + items).str.strip()


# --------------------------------------------------
# 4. Main evaluation
# --------------------------------------------------

def main():
    print("=" * 60)
    print("FINAL LINEAR SVM MODEL EVALUATION")
    print("=" * 60)

    train_df = load_dataset(TRAIN_PATH)
    validation_df = load_dataset(VALIDATION_PATH)
    test_df = load_dataset(TEST_PATH)

    # Combine training and validation data after model selection.
    train_validation_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    X_train = prepare_text(train_validation_df)
    y_train = train_validation_df["expense_category"].astype(str)

    X_test = prepare_text(test_df)
    y_test = test_df["expense_category"].astype(str)

    # Build a single pipeline so the vectorizer and classifier
    # are saved together.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("Training records:", len(train_validation_df))
    print("Test records:", len(test_df))

    print("\nTraining final model...")
    model.fit(X_train, y_train)

    # Save model for later use in the Streamlit application.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Saved model to:", MODEL_PATH)

    # Evaluate only after model selection is complete.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=sorted(y_train.unique()),
        zero_division=0,
    ))

    labels = sorted(y_train.unique())
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    confusion_path = RESULTS_DIR / "test_confusion_matrix.csv"
    confusion_df.to_csv(confusion_path)

    metrics_df = pd.DataFrame([{
        "model": "Linear SVM",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_records": len(train_validation_df),
        "test_records": len(test_df),
    }])

    metrics_path = RESULTS_DIR / "final_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nConfusion matrix saved to:", confusion_path)
    print("Test metrics saved to:", metrics_path)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()

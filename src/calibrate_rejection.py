
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "data" / "splits"
RESULTS_DIR = BASE_DIR / "data" / "model_results"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"

RESULTS_PATH = RESULTS_DIR / "rejection_calibration.csv"

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------

def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def prepare_text(df):
    merchant = df["merchant_name"].fillna("").astype(str)
    items = df["purchased_items"].fillna("").astype(str)

    return (merchant + " " + items).str.strip()


def main():
    train_df = load_dataset(TRAIN_PATH)
    validation_df = load_dataset(VALIDATION_PATH)

    X_train = prepare_text(train_df)
    y_train = train_df["expense_category"].astype(str)

    X_val = prepare_text(validation_df)
    y_val = validation_df["expense_category"].astype(str)

    # Train only on the training split.
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

    model.fit(X_train, y_train)

    # Get decision scores for validation records.
    scores = model.decision_function(X_val)

    # For multiclass LinearSVC, each column corresponds to a class.
    ranked_scores = pd.DataFrame(scores).apply(
        lambda row: sorted(row, reverse=True),
        axis=1,
        result_type="expand",
    )

    top_score = ranked_scores[0]
    second_score = ranked_scores[1]
    margins = top_score - second_score

    predictions = model.predict(X_val)
    correct = (predictions == y_val.to_numpy())

    # Explore thresholds based on observed validation margins.
    thresholds = sorted(set(
        [0.0]
        + margins.quantile(
            [0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90]
        ).round(6).tolist()
    ))

    results = []

    for threshold in thresholds:
        accepted = margins >= threshold

        coverage = accepted.mean()

        selective_accuracy = (
            accuracy_score(
                y_val.to_numpy()[accepted],
                predictions[accepted],
            )
            if accepted.any()
            else None
        )

        accepted_correct = (
            correct[accepted].mean()
            if accepted.any()
            else None
        )

        results.append({
            "margin_threshold": threshold,
            "coverage": coverage,
            "accepted_accuracy": selective_accuracy,
            "accepted_correct_rate": accepted_correct,
            "accepted_records": int(accepted.sum()),
            "rejected_records": int((~accepted).sum()),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("Validation records:", len(validation_df))
    print("Accuracy without rejection:",
          accuracy_score(y_val, predictions))
    print("\nCalibration results:")
    print(results_df.to_string(index=False))
    print("\nSaved results to:", RESULTS_PATH)


if __name__ == "__main__":
    main()

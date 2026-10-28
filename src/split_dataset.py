
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "receipt_dataset_ml_ready.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "splits"

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def main():
    if not INPUT_PATH.exists():
        print("ERROR: ML-ready dataset not found.")
        print("Expected:", INPUT_PATH)
        return

    df = pd.read_excel(INPUT_PATH)

    required = {"receipt_id", "expense_category"}
    missing = required - set(df.columns)

    if missing:
        print("ERROR: Missing columns:", sorted(missing))
        return

    if df["receipt_id"].duplicated().any():
        print("ERROR: Duplicate receipt IDs found.")
        return

    if df["expense_category"].isna().any():
        print("ERROR: Missing category labels found.")
        return

    category_counts = df["expense_category"].value_counts()

    print("=" * 55)
    print("DATASET SPLITTING")
    print("=" * 55)
    print("Total records:", len(df))
    print("\nOriginal category counts:")
    print(category_counts.to_string())

    # First, reserve 15% for the final test set.
    train_validation, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["expense_category"],
    )

    # Split the remaining 85% so validation is 15% of the full dataset.
    validation_fraction = VALIDATION_SIZE / (
        TRAIN_SIZE + VALIDATION_SIZE
    )

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=train_validation["expense_category"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nSplit results:")
    for name, subset in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test),
    ]:
        print(f"\n{name}: {len(subset)} records")
        print(
            subset["expense_category"]
            .value_counts()
            .reindex(category_counts.index, fill_value=0)
            .to_string()
        )

    print("\nFiles saved to:", OUTPUT_DIR)
    print("The original dataset was not changed.")


if __name__ == "__main__":
    main()


import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "receipt_dataset_cleaned.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "receipt_dataset_ml_ready.xlsx"

VALID_CATEGORIES = {
    "Food and Beverage",
    "Groceries",
    "Transportation",
    "Shopping",
    "Health & Personal Care",
}

# Confirmed duplicate IDs: retain the first record in each pair.
DUPLICATE_IDS_TO_EXCLUDE = {"R0041", "R0074"}

# Exclude these records from training for now.
# R0031 is a future voucher.
# R0104 has an unconfirmed payment amount.
RECORD_IDS_TO_EXCLUDE = {"R0031", "R0104"}


def main():
    if not INPUT_PATH.exists():
        print("ERROR: Cleaned dataset not found:", INPUT_PATH)
        return

    df = pd.read_excel(INPUT_PATH)

    required_columns = {
        "receipt_id",
        "merchant_name",
        "purchased_items",
        "total_amount",
        "expense_category",
        "record_type",
    }

    missing = required_columns - set(df.columns)

    if missing:
        print("ERROR: Missing required columns:", sorted(missing))
        return

    # Normalise text used in eligibility checks.
    categories = df["expense_category"].astype("string").str.strip()
    record_types = df["record_type"].astype("string").str.strip()
    receipt_ids = df["receipt_id"].astype("string").str.strip()
    amounts = pd.to_numeric(df["total_amount"], errors="coerce")

    # Start with records suitable for completed-purchase classification.
    eligible = (
        categories.isin(VALID_CATEGORIES)
        & record_types.eq("Purchase")
        & amounts.notna()
        & df["merchant_name"].notna()
        & df["merchant_name"].astype(str).str.strip().ne("")
        & df["purchased_items"].notna()
        & df["purchased_items"].astype(str).str.strip().ne("")
    )

    # Exclude confirmed duplicates and records unsuitable for training.
    eligible &= ~receipt_ids.isin(
        DUPLICATE_IDS_TO_EXCLUDE | RECORD_IDS_TO_EXCLUDE
    )

    ml_df = df.loc[eligible].copy()

    # Ensure category labels are standardised in the ML dataset.
    ml_df["expense_category"] = (
        ml_df["expense_category"].astype("string").str.strip()
    )

    ml_df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")

    print("=" * 55)
    print("ML DATASET PREPARATION")
    print("=" * 55)
    print("Original cleaned records:", len(df))
    print("Eligible ML records:", len(ml_df))
    print("Excluded records:", len(df) - len(ml_df))
    print("\nCategory distribution:")
    print(ml_df["expense_category"].value_counts().to_string())
    print("\nSaved to:", OUTPUT_PATH)
    print("\nThe cleaned master dataset remains unchanged.")


if __name__ == "__main__":
    main()

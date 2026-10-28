
import pandas as pd
from pathlib import Path

# Locate the cleaned Excel dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "receipt_dataset_cleaned.xlsx"
REPORT_PATH = BASE_DIR / "data" / "audit_report.xlsx"

EXPECTED_CATEGORIES = {
    "Food and Beverage",
    "Groceries",
    "Transportation",
    "Shopping",
    "Health & Personal Care",
    "Needs Review",
}

ELIGIBLE_RECORD_TYPES = {"Purchase", "Reward Redemption"}

REQUIRED_COLUMNS = [
    "receipt_id",
    "merchant_name",
    "transaction_date",
    "transaction_time",
    "purchased_items",
    "total_amount",
    "currency",
    "expense_category",
    "record_type",
]


def main():
    if not DATA_PATH.exists():
        print("ERROR: Cleaned dataset not found.")
        print("Expected location:", DATA_PATH)
        print("Please run data_cleaning.py first.")
        return

    df = pd.read_excel(DATA_PATH)

    print("=" * 65)
    print("RECEIPT DATASET AUDIT")
    print("=" * 65)
    print(f"File: {DATA_PATH.name}")
    print(f"Number of records: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\n1. COLUMN CHECK")
    print("-" * 45)

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
        print("Please fix the column names before continuing.")
        return

    print("All required columns are present.")

    print("\nColumns found:")
    for col in df.columns:
        print("-", col)

    # Standardise text for checks without changing the source dataset.
    categories = df["expense_category"].astype("string").str.strip()
    record_types = df["record_type"].astype("string").str.strip()
    amounts = pd.to_numeric(df["total_amount"], errors="coerce")

    print("\n2. MISSING VALUES")
    print("-" * 45)

    missing_counts = df[REQUIRED_COLUMNS].isna().sum()
    print(
        missing_counts[missing_counts > 0].to_string()
        if missing_counts.any()
        else "No missing values in required columns."
    )

    print("\n3. EXPENSE CATEGORY CHECK")
    print("-" * 45)

    invalid_categories = (
        categories.isna() | ~categories.isin(EXPECTED_CATEGORIES)
    )

    if invalid_categories.any():
        print("Categories needing review:")
        print(
            df.loc[invalid_categories, [
                "receipt_id", "merchant_name",
                "expense_category", "record_type"
            ]].to_string(index=False)
        )
    else:
        print("All categories match the expected list.")

    print("\nCategory distribution (all records):")
    print(
        categories.value_counts(dropna=False).to_string()
    )

    print("\n4. RECORD TYPE CHECK")
    print("-" * 45)

    missing_record_types = record_types.isna() | record_types.eq("")
    print("Missing record types:", missing_record_types.sum())
    print("\nRecord type distribution:")
    print(record_types.value_counts(dropna=False).to_string())

    print("\n5. AMOUNT CHECK")
    print("-" * 45)

    non_numeric_amounts = (
        df["total_amount"].notna() & amounts.isna()
    )
    missing_amounts = (
        df["total_amount"].isna()
        | df["total_amount"].astype(str).str.strip().eq("")
    )
    zero_or_negative = amounts.notna() & (amounts <= 0)

    print("Missing amounts:", missing_amounts.sum())
    print("Non-numeric amounts:", non_numeric_amounts.sum())
    print("Zero or negative amounts:", zero_or_negative.sum())

    if missing_amounts.any():
        print("\nRecords with missing amounts:")
        print(
            df.loc[missing_amounts, [
                "receipt_id", "merchant_name",
                "total_amount", "record_type"
            ]].to_string(index=False)
        )

    if zero_or_negative.any():
        print("\nZero or negative amounts (review in context):")
        print(
            df.loc[zero_or_negative, [
                "receipt_id", "merchant_name",
                "total_amount", "record_type"
            ]].to_string(index=False)
        )

    print("\n6. POSSIBLE DUPLICATES")
    print("-" * 45)

    duplicate_fields = [
        "merchant_name",
        "transaction_date",
        "total_amount",
        "purchased_items",
    ]

    duplicate_mask = df.duplicated(
        subset=duplicate_fields,
        keep=False,
    )

    if duplicate_mask.any():
        print(
            f"{duplicate_mask.sum()} records belong to "
            "possible duplicate groups."
        )
        duplicate_columns = [
            "receipt_id", "source_pdf", "page_number",
            "merchant_name", "transaction_date",
            "transaction_time", "total_amount",
            "purchased_items",
        ]
        duplicate_columns = [
            col for col in duplicate_columns if col in df.columns
        ]
        print(
            df.loc[duplicate_mask, duplicate_columns]
            .sort_values(["merchant_name", "transaction_date"])
            .to_string(index=False)
        )
        print("These are candidates, not confirmed duplicates.")
    else:
        print("No possible duplicates found.")

    print("\n7. MACHINE LEARNING READINESS")
    print("-" * 45)

    # Purchases and reward redemptions can be considered for
    # classification only if the category and text are reliable.
    # Vouchers and other record types are excluded from this candidate pool.
    candidate_mask = (
        record_types.isin(ELIGIBLE_RECORD_TYPES)
        & categories.isin(EXPECTED_CATEGORIES - {"Needs Review"})
        & df["merchant_name"].notna()
        & df["merchant_name"].astype(str).str.strip().ne("")
        & df["purchased_items"].notna()
        & df["purchased_items"].astype(str).str.strip().ne("")
        & amounts.notna()
        & ~non_numeric_amounts
    )

    candidates = df.loc[candidate_mask].copy()

    print("Potential ML records:", len(candidates))
    print("Records excluded from candidate pool:", len(df) - len(candidates))
    print("\nPotential ML records by category:")
    print(
        candidates["expense_category"]
        .astype("string")
        .str.strip()
        .value_counts()
        .reindex(sorted(EXPECTED_CATEGORIES - {"Needs Review"}), fill_value=0)
        .to_string()
    )

    print(
        "\nNOTE: This is a preliminary eligibility check, "
        "not final approval for training."
    )
    print(
        "Manually reviewed duplicates, non-receipt slips, "
        "and uncertain transactions still need appropriate decisions."
    )

    print("\n8. REVIEW SUMMARY")
    print("-" * 45)

    review = pd.DataFrame({
        "receipt_id": df["receipt_id"],
        "merchant_name": df["merchant_name"],
        "transaction_date": df["transaction_date"],
        "transaction_time": df["transaction_time"],
        "purchased_items": df["purchased_items"],
        "total_amount": df["total_amount"],
        "currency": df["currency"],
        "expense_category": df["expense_category"],
        "record_type": df["record_type"],
    })

    review["review_reason"] = ""

    review.loc[
        df["merchant_name"].isna()
        | df["merchant_name"].astype(str).str.strip().eq(""),
        "review_reason"
    ] += "Missing merchant; "

    review.loc[
        df["purchased_items"].isna()
        | df["purchased_items"].astype(str).str.strip().eq(""),
        "review_reason"
    ] += "Missing items; "

    review.loc[
        invalid_categories, "review_reason"
    ] += "Invalid or missing category; "

    review.loc[
        non_numeric_amounts, "review_reason"
    ] += "Non-numeric amount; "

    review.loc[
        missing_amounts, "review_reason"
    ] += "Missing amount; "

    review.loc[
        missing_record_types, "review_reason"
    ] += "Missing record type; "

    review.loc[
        duplicate_mask, "review_reason"
    ] += "Possible duplicate; verify manually; "

    review.loc[
        record_types.eq("Voucher").fillna(False),
        "review_reason"
    ] += "Voucher excluded from ML candidate pool; "

    review.loc[
        ~record_types.isin(ELIGIBLE_RECORD_TYPES),
        "review_reason"
    ] += "Record type not currently eligible for ML; "

    review = review[review["review_reason"] != ""]

    print("Records flagged for review:", len(review))

    # Save the audit report separately; source files remain unchanged.
    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        review.to_excel(
            writer, sheet_name="Records to Review", index=False
        )
        df.loc[invalid_categories].to_excel(
            writer, sheet_name="Invalid Categories", index=False
        )
        df.loc[duplicate_mask].to_excel(
            writer, sheet_name="Possible Duplicates", index=False
        )
        candidates.to_excel(
            writer, sheet_name="ML Candidates", index=False
        )

    print("\nAudit report saved to:")
    print(REPORT_PATH)
    print("\nAudit completed. The cleaned dataset was not changed.")


if __name__ == "__main__":
    main()

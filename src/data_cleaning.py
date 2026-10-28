
import pandas as pd
from pathlib import Path

# =========================================================
# 1. FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "receipt_dataset.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "receipt_dataset_cleaned.xlsx"


# =========================================================
# 2. VALID EXPENSE CATEGORIES
# =========================================================

VALID_CATEGORIES = {
    "Food and Beverage",
    "Groceries",
    "Transportation",
    "Shopping",
    "Health & Personal Care",
    "Needs Review",
}


# =========================================================
# 3. MAIN CLEANING FUNCTION
# =========================================================

def main():

    # Check whether the input file exists
    if not INPUT_PATH.exists():
        print("ERROR: Dataset not found!")
        print(f"Expected location: {INPUT_PATH}")
        return

    # Read the original Excel dataset
    df = pd.read_excel(INPUT_PATH)

    print("=" * 60)
    print("EXPENSE DATASET CLEANING")
    print("=" * 60)

    print(f"Input file: {INPUT_PATH.name}")
    print(f"Original records: {len(df)}")

    # -----------------------------------------------------
    # STEP 1: CHECK REQUIRED COLUMNS
    # -----------------------------------------------------

    required_columns = [
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

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Required columns are missing:")
        print(missing_columns)
        print(
            "\nPlease check your Excel column names. "
            "The original file has not been changed."
        )
        return

    # -----------------------------------------------------
    # STEP 2: CLEAN TEXT SPACING
    # -----------------------------------------------------

    text_columns = [
        "receipt_id",
        "source_pdf",
        "merchant_name",
        "purchased_items",
        "currency",
        "expense_category",
        "payment_method",
        "verification_status",
        "extraction_notes",
        "record_type",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda value: value.strip()
                if isinstance(value, str)
                else value
            )

    print("\nText spacing standardised.")

    # -----------------------------------------------------
    # STEP 3: STANDARDISE DATES
    # -----------------------------------------------------
    # Assumption: dates currently use MM/DD/YYYY.
    # Output format: DD/MM/YYYY.
    #
    # Unparseable dates are preserved for manual review.

    original_dates = df["transaction_date"].copy()

    parsed_dates = pd.to_datetime(
        original_dates,
        format="%m/%d/%Y",
        errors="coerce"
    )

    df["transaction_date"] = parsed_dates.dt.strftime(
        "%d/%m/%Y"
    )

    failed_dates = (
        parsed_dates.isna()
        & original_dates.notna()
        & original_dates.astype(str).str.strip().ne("")
    )

    df.loc[failed_dates, "transaction_date"] = (
        original_dates.loc[failed_dates]
    )

    df["cleaning_notes"] = ""

    df.loc[
        failed_dates,
        "cleaning_notes"
    ] += "Date not parsed; check original receipt; "

    print("Dates processed using MM/DD/YYYY input format.")
    print("Output date format: DD/MM/YYYY.")
    print(f"Dates requiring review: {failed_dates.sum()}")

    # -----------------------------------------------------
    # STEP 4: CONVERT AMOUNTS TO NUMERIC VALUES
    # -----------------------------------------------------

    original_amounts = df["total_amount"].copy()

    numeric_amounts = pd.to_numeric(
        original_amounts,
        errors="coerce"
    )

    # Identify non-empty values that could not be converted
    invalid_amounts = (
        original_amounts.notna()
        & original_amounts.astype(str).str.strip().ne("")
        & numeric_amounts.isna()
    )

    df["total_amount"] = numeric_amounts

    df.loc[
        invalid_amounts,
        "cleaning_notes"
    ] += "Non-numeric amount; check original receipt; "

    # Flag missing amounts
    missing_amounts = (
        original_amounts.isna()
        | original_amounts.astype(str).str.strip().eq("")
    )

    df.loc[
        missing_amounts,
        "cleaning_notes"
    ] += "Missing amount; "

    # Flag zero and negative amounts for review.
    # These may be valid reward redemptions, refunds, or adjustments.

    numeric_values = pd.to_numeric(
        df["total_amount"],
        errors="coerce"
    )

    zero_or_negative = (
        numeric_values.notna()
        & (numeric_values <= 0)
    )

    df.loc[
        zero_or_negative,
        "cleaning_notes"
    ] += "Zero or negative amount; confirm transaction type; "

    print("\nAmount processing completed.")
    print(f"Non-numeric amounts: {invalid_amounts.sum()}")
    print(f"Missing amounts: {missing_amounts.sum()}")
    print(f"Zero or negative amounts: {zero_or_negative.sum()}")

    # -----------------------------------------------------
    # STEP 5: CHECK EXPENSE CATEGORIES
    # -----------------------------------------------------

    categories = df["expense_category"].astype("string").str.strip()

    invalid_categories = (
        categories.isna()
        | ~categories.isin(VALID_CATEGORIES)
    )

    df.loc[
        invalid_categories,
        "cleaning_notes"
    ] += "Invalid or missing category; review required; "

    print("\nCategory check completed.")
    print(f"Categories requiring review: {invalid_categories.sum()}")

    # -----------------------------------------------------
    # STEP 6: CHECK RECORD TYPES
    # -----------------------------------------------------
    # Existing record_type values are preserved.
    # Blank values are flagged rather than guessed.

    record_types = df["record_type"].astype("string").str.strip()

    missing_record_types = (
        record_types.isna()
        | record_types.eq("")
    )

    df.loc[
        missing_record_types,
        "cleaning_notes"
    ] += "Missing record_type; classify manually; "

    print("\nRecord type check completed.")
    print(f"Missing record types: {missing_record_types.sum()}")

    # -----------------------------------------------------
    # STEP 7: CHECK MISSING MERCHANTS AND ITEMS
    # -----------------------------------------------------

    for column in ["merchant_name", "purchased_items"]:

        missing_values = (
            df[column].isna()
            | df[column].astype(str).str.strip().eq("")
        )

        df.loc[
            missing_values,
            "cleaning_notes"
        ] += f"Missing {column}; "

        print(
            f"Missing {column}: {missing_values.sum()}"
        )

    # -----------------------------------------------------
    # STEP 8: FLAG POSSIBLE DUPLICATES
    # -----------------------------------------------------
    # Do not remove duplicates automatically.

    duplicate_fields = [
        "merchant_name",
        "transaction_date",
        "total_amount",
        "purchased_items",
    ]

    possible_duplicates = df.duplicated(
        subset=duplicate_fields,
        keep=False
    )

    df.loc[
        possible_duplicates,
        "cleaning_notes"
    ] += "Possible duplicate; verify before exclusion; "

    print("\nDuplicate check completed.")
    print(
        "Records in possible duplicate groups:",
        possible_duplicates.sum()
    )

    # -----------------------------------------------------
    # STEP 9: SAVE CLEANED COPY
    # -----------------------------------------------------

    df.to_excel(
        OUTPUT_PATH,
        index=False,
        engine="openpyxl"
    )

    print("\n" + "=" * 60)
    print("CLEANING COMPLETED")
    print("=" * 60)

    print(f"Original records: {len(pd.read_excel(INPUT_PATH))}")
    print(f"Output records: {len(df)}")
    print(f"Output file: {OUTPUT_PATH}")
    print("\nOriginal dataset remains unchanged.")
    print("Review cleaning_notes before using the dataset for ML.")


# =========================================================
# 4. RUN SCRIPT
# =========================================================

if __name__ == "__main__":
    main()

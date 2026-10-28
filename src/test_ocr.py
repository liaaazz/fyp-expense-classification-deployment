from pathlib import Path

import pytesseract
from PIL import Image, ImageOps, ImageFilter

# Configure Tesseract on Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

IMAGE_PATH = Path(
    r"C:\Users\user\Desktop\fyp_expense_classification\data\raw_receipts\photo_2026-10-11_17-11-32.jpg"
)




def preprocess_receipt(image):
    image = ImageOps.grayscale(image)
    image = image.resize(
        (image.width * 2, image.height * 2),
        Image.Resampling.LANCZOS,
    )
    image = ImageOps.autocontrast(image, cutoff=2)
    image = image.filter(ImageFilter.SHARPEN)
    return image




def main():
    if not IMAGE_PATH.exists():
        print("Receipt image not found:", IMAGE_PATH)
        print("Update IMAGE_PATH to your actual receipt image.")
        return

    original = Image.open(IMAGE_PATH).convert("RGB")

    # Crop the blank upper area and side edges.
    original = original.crop((
        40,
        250,
        original.width - 40,
        original.height,
    ))

    processed = preprocess_receipt(original)

    # Save the processed image.
    output_path = IMAGE_PATH.parent / "receipt_processed.png"
    processed.save(output_path)

    # Perform OCR on the processed image.
    extracted_text = pytesseract.image_to_string(
        processed,
        config="--psm 6",
        lang="eng",
    ).strip()

    print("=" * 50)
    print("PREPROCESSED OCR RESULT")
    print("=" * 50)
    print(extracted_text or "No text detected.")

    print("\nProcessed image saved to:")
    print(output_path)


if __name__ == "__main__":
    main()

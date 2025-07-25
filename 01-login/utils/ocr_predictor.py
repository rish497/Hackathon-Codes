# ocr_predictor.py
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_file(file_path):
    text = ""
    ext = os.path.splitext(file_path)[-1].lower()

    try:
        if ext == ".pdf":
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        elif ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
    except Exception as e:
        text = f"Error extracting text: {str(e)}"

    return text

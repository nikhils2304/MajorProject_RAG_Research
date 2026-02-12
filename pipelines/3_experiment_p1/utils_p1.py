from pathlib import Path
from PyPDF2 import PdfReader
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_p1.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path, language: str = "English") -> str:
    """
    Extracts text from a PDF file using language-specific methods.
    Args:
        pdf_path (Path): Path to the PDF file.
        language (str): Language of the document.
    Returns:
        str: Extracted text content.
    """
    try:
        if language == "Tamil":
            # Use pdfplumber for Tamil
            with pdfplumber.open(pdf_path) as pdf:
                text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                return "\n".join(text)
        
        elif language == "Bengali":
            # Use Tesseract OCR for Bengali
            images = convert_from_path(str(pdf_path))
            text = []
            for image in images:
                page_text = pytesseract.image_to_string(image, lang='ben')
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
            
        else:
            # Default PyPDF2
            reader = PdfReader(pdf_path)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
            
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return ""

def get_pdf_files(directory: Path) -> list:
    """Returns a list of PDF files in the directory."""
    return list(directory.rglob("*.pdf"))

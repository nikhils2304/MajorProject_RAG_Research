
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: Path):
    """Sets up a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to create log file handler: {e}")
            
            
    return logger

import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load Env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
translation_model = genai.GenerativeModel('gemini-2.0-flash')

def translate_text(text: str, target_lang: str = "English") -> str:
    """
    Translates text to target language using Gemini 2.0 Flash.
    Retry logic included.
    """
    if not text.strip(): return ""
    
    prompt = f"Translate the following text to {target_lang}. Return ONLY the translation, no preamble.\n\nText:\n{text}"
    
    for attempt in range(3):
        try:
            response = translation_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Translation failed (Attempt {attempt+1}): {e}")
            time.sleep(2 * (attempt + 1))
            
    return text # Return original if failed

# Reused extraction logic


# Reused extraction logic
from pathlib import Path
from PyPDF2 import PdfReader
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

def get_pdf_files(directory: Path) -> list:
    """Returns a list of PDF files in the directory sorted alphabetically."""
    return sorted(list(directory.rglob("*.pdf")))

def extract_text_from_pdf(pdf_path: Path, language: str = "English") -> str:
    """
    Extracts text from a PDF file using language-specific methods.
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
            # Default PyPDF2 (English, Hindi)
            reader = PdfReader(pdf_path)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
            
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


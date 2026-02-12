import os
import json
import json_repair
import logging
from pathlib import Path
from PyPDF2 import PdfReader
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import google.genai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_gemini(api_key):
    """
    Initialize the Gemini client.
    
    Args:
        api_key (str): Google API Key.
        
    Returns:
        genai.Client: Initialized Gemini client.
        
    Raises:
        ValueError: If API key is not provided.
    """
    if not api_key:
        logger.error("Google API Key not provided.")
        raise ValueError("Google API Key not provided.")
    return genai.Client(api_key=api_key)

def extract_text_from_pdf(file_path, language="English"):
    """
    Extract text content from a PDF file using language-specific methods.
    
    Args:
        file_path (str or Path): Path to the PDF file.
        language (str): Language of the document (affects extraction method).
        
    Returns:
        str: Extracted text content, or empty string on failure.
    """
    try:
        if language == "Tamil":
            # Use pdfplumber for Tamil (handle fragmented characters)
            with pdfplumber.open(file_path) as pdf:
                text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                return "\n\n".join(text).strip()
        
        elif language == "Bengali":
            # Use Tesseract OCR for Bengali (scanned/image PDFs)
            images = convert_from_path(str(file_path))
            text = []
            for image in images:
                page_text = pytesseract.image_to_string(image, lang='ben')
                if page_text:
                    text.append(page_text)
            return "\n\n".join(text).strip()
            
        else:
            # Default PyPDF2 for English/Hindi
            reader = PdfReader(str(file_path))
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n\n".join(text).strip()
            
    except Exception as e:
        logger.error(f"Error extraction text from {Path(file_path).name} ({language}): {e}")
        return ""

def load_json_response(response_text):
    """
    Parse JSON from LLM response, handling potential formatting issues.
    
    Args:
        response_text (str): Raw text response from LLM.
        
    Returns:
        dict or list: Parsed JSON data, or None on failure.
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            return json_repair.loads(response_text)
        except Exception as e:
            logger.error(f"JSON Parsing failed: {e}")
            return None

def save_json(data, filepath):
    """
    Save data to a JSON file.
    
    Args:
        data (any): Serializable data to save.
        filepath (str or Path): Destination file path.
    """
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} items to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")

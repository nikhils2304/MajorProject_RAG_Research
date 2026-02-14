
import sys
import logging
from pathlib import Path
import google.generativeai as genai
import os
import time

# Add P7 to path to reuse translation logic if needed, 
# but better to copy/paste the critical parts to avoid import mess if P7 has specific relative imports.
# Actually, P7 utils are simple. Let's just reimplement the necessary parts or import cleanly.

sys.path.append(str(Path(__file__).parent))
import config

def setup_logger(name: str, log_file: Path, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

# Configure Gemini for Translation
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
translation_model = genai.GenerativeModel('gemini-2.0-flash')

def translate_text(text: str, target_lang: str = "English") -> str:
    """
    Translates text to target language using Gemini 2.0 Flash.
    Retry logic included.
    """
    if not text or not text.strip(): return ""
    
    prompt = f"Translate the following text to {target_lang}. Return ONLY the translation, no preamble.\n\nText:\n{text}"
    
    for attempt in range(3):
        try:
            response = translation_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Translation failed (Attempt {attempt+1}): {e}")
            time.sleep(2 * (attempt + 1))
            
    return text # Return original if failed

def get_pdf_files(directory: Path):
    return list(directory.glob("*.pdf"))

def load_translated_docs(language: str):
    """
    Loads translated text files from P7's translated_docs directory.
    Returns list of Document objects (compatible with LangChain).
    """
    from langchain_core.documents import Document
    
    # Path to P7 translated docs
    # Note: Logic from P7 was: 
    # lang_out_dir = config.P7_DIR / "translated_docs" / language
    
    source_dir = config.TRANSLATED_DOCS_DIR / language
    docs = []
    
    if not source_dir.exists():
        print(f"Warning: Directory not found {source_dir}")
        return []
        
    for txt_file in source_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Metadata - try to reconstruct or keep simple
        # P7 index script kept source filename.
        # We need this for evaluation (matching 'source_file' in QA pair).
        # txt_file.stem should be original pdf stem.
        original_pdf_name = txt_file.stem + ".pdf" 
        
        docs.append(Document(
            page_content=content,
            metadata={"source": original_pdf_name, "language": language}
        ))
    return docs


import sys
import time
from pathlib import Path
from tqdm import tqdm

# Add pipeline dir to path
sys.path.append(str(Path(__file__).parent))

import config
import utils_p7

logger = utils_p7.setup_logger("P7_Translator", config.PIPELINE_DIR / "translation.log")

def process_language(language: str):
    logger.info(f"--- Processing {language} ---")
    print(f"DEBUG: Processing {language}")
    
    # 1. Get Docs

    dataset_folder = config.LANG_TO_PDF_DIR.get(language)
    if not dataset_folder:
        logger.error(f"No folder for {language}")
        return
        
    data_path = config.DATASET_DIR / dataset_folder
    all_pdfs = utils_p7.get_pdf_files(data_path)
    
    # 2. Select Top-25
    selected_pdfs = all_pdfs[:config.DOCS_PER_LANG]
    logger.info(f"Selected {len(selected_pdfs)} / {len(all_pdfs)} documents.")
    
    # 3. Output Dir
    lang_out_dir = config.PIPELINE_DIR / "translated_docs" / language
    print(f"DEBUG: Creating dir: {lang_out_dir}")
    lang_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Translate Loop
    for pdf_file in tqdm(selected_pdfs, desc=f"Translating {language}"):
        # Check if already done
        out_file = lang_out_dir / (pdf_file.stem + ".txt")
        # FORCE OVERWRITE FOR DEBUG
        # if out_file.exists():
        #     continue
            
        # Extract
        text = utils_p7.extract_text_from_pdf(pdf_file, language=language)
        if not text:
            logger.warning(f"Empty text for {pdf_file.name}")
            continue
            
        # Translate (if not English)
        if language == "English":
            translated_text = text # Identity
        else:
            # Use Gemini via utils_p7
            translated_text = utils_p7.translate_text(text, target_lang="English")
            time.sleep(0.5) # Rate limit politeness
            
        # Save
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(translated_text)
            
    logger.info(f"Finished {language}.")

if __name__ == "__main__":
    languages = ["English", "Hindi", "Tamil", "Bengali"]
    for lang in languages:
        process_language(lang)

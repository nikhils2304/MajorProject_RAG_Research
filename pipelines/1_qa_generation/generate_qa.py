import os
import argparse
import time
import random
import logging
from pathlib import Path
from tqdm import tqdm
import config
import prompts
import utils
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_qa_for_file(client, file_path, language, num_pairs=3):
    """
    Generate QA pairs for a single PDF file using Gemini.
    
    Args:
        client: Gemini API client.
        file_path (Path): Path to the source PDF.
        language (str): Target language for QA pairs.
        num_pairs (int): Number of QA pairs to generate.
        
    Returns:
        list: List of generated QA dictionaries.
    """
    text = utils.extract_text_from_pdf(file_path, language=language)
    if not text:
        logger.warning(f"No text extracted from {file_path}")
        return []
    
    # Truncate text to safe limit (~30k chars)
    text = text[:30000]

    prompt = prompts.QA_GENERATION_PROMPT.format(language=language, num_pairs=num_pairs)

    try:
        response = client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=[prompt, text],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
                response_mime_type="application/json"
            )
        )
        
        qa_pairs = utils.load_json_response(response.text)
        if not qa_pairs:
            logger.warning(f"Empty or invalid JSON response for {file_path}")
            return []

        # Add metadata
        for pair in qa_pairs:
            pair['source_file'] = Path(file_path).name
            pair['source_path'] = str(file_path)
            pair['language'] = language
            
        return qa_pairs

    except Exception as e:
        logger.error(f"Error generating QA for {file_path}: {e}")
        return []

def critique_and_filter(client, qa_pairs, language):
    """
    Critique QA pairs and filter based on quality score.
    
    Args:
        client: Gemini API client.
        qa_pairs (list): List of QA pair dictionaries.
        language (str): Language of content.
        
    Returns:
        list: Filtered list of high-quality QA pairs.
    """
    filtered_pairs = []
    
    logger.info(f"Critiquing {len(qa_pairs)} pairs...")
    
    for pair in tqdm(qa_pairs, desc="Critique Progress"):
        prompt = prompts.CRITIQUE_PROMPT.format(
            language=language,
            question=pair.get('question', ''),
            answer=pair.get('answer', ''),
            context=pair.get('context', '')
        )
        
        try:
            response = client.models.generate_content(
                model=config.GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                    response_mime_type="application/json"
                )
            )
            
            critique = utils.load_json_response(response.text)
            
            if critique:
                pair.update(critique)
                
                scores = [
                    critique.get('groundedness', 0),
                    critique.get('relevance', 0),
                    critique.get('standalone', 0),
                    critique.get('quality', 0)
                ]
                
                if all(s >= config.MIN_QUALITY_SCORE for s in scores):
                    filtered_pairs.append(pair)
            
            time.sleep(config.REQUEST_DELAY)

        except Exception as e:
            logger.error(f"Critique error: {e}")
            continue
            
    return filtered_pairs

def main():
    parser = argparse.ArgumentParser(description="Multilingual QA Generation Pipeline")
    parser.add_argument("--language", type=str, required=True, choices=["English", "Hindi", "Tamil", "Bengali"], help="Target language")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    args = parser.parse_args()

    try:
        client = utils.init_gemini(config.GOOGLE_API_KEY)
    except ValueError as e:
        logger.critical(str(e))
        return

    dataset_dir = Path(config.DATASET_ROOT).resolve()
    dir_map = {
        "English": "EnglishDataset",
        "Hindi": "HindiDataset",
        "Tamil": "TamilDataset",
        "Bengali": "BengaliDataset"
    }
    
    lang_dir = dataset_dir / dir_map[args.language]
    
    if not lang_dir.exists():
        logger.error(f"Directory not found: {lang_dir}")
        return

    pdf_files = list(lang_dir.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files for {args.language}")
    
    if args.limit:
        pdf_files = pdf_files[:args.limit]
        logger.info(f"Limiting processing to first {args.limit} files.")

    all_qa_pairs = []

    for pdf_file in tqdm(pdf_files, desc="Generation Progress"):
        pairs = generate_qa_for_file(client, pdf_file, args.language, num_pairs=config.QA_PAIRS_PER_FILE)
        all_qa_pairs.extend(pairs)
        time.sleep(config.REQUEST_DELAY)

    logger.info(f"Generated {len(all_qa_pairs)} raw QA pairs.")

    if all_qa_pairs:
        filtered_pairs = critique_and_filter(client, all_qa_pairs, args.language)
        
        logger.info(f"Filtered down to {len(filtered_pairs)} high-quality pairs.")

        output_file = Path(config.OUTPUT_ROOT) / f"{args.language.lower()}_qa.json"
        utils.save_json(filtered_pairs, output_file)
    else:
        logger.warning("No pairs generated.")

if __name__ == "__main__":
    main()

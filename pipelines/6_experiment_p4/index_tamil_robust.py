
import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
import re
import signal

# Add config path
sys.path.append(str(Path(__file__).parent))

import config
import utils_p4

# Add P1 for embeddings
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))
from embeddings import MonolingualEmbedding

logger = utils_p4.logger

def sanitize_text(text):
    """Removes null bytes and other common problematic control characters."""
    if not text:
        return ""
    # Remove null bytes
    text = text.replace('\x00', '')
    # Replace other non-printable characters (optional, but good for safety)
    # text = re.sub(r'[^\x20-\x7E\n\t\r]', '', text) # This might remove Tamil chars! Don't do this.
    return text

import argparse

def index_tamil_robust(start_index=0):
    language = "Tamil"
    logger.info(f"--- Starting Robust Indexing for {language} (Start Index: {start_index}) ---")
    
    # 1. Setup paths
    dataset_folder = config.LANG_TO_FOLDER.get(language)
    data_path = config.DATASET_DIR / dataset_folder
    persist_dir = config.CHROMA_DB_DIR / f"{language.lower()}_index"
    
    # 2. Initialize Model
    model_name = config.MODELS[language]
    logger.info(f"Loading Model: {model_name}")
    embedding_fn = MonolingualEmbedding(model_name=model_name)

    # 3. Initialize Vector Store (Persistent)
    vector_store = Chroma(
        collection_name=f"{language.lower()}_collection",
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir)
    )

    # 4. Load Documents and Index ONE BY ONE
    pdf_files = sorted(utils_p4.get_pdf_files(data_path))
    logger.info(f"Found {len(pdf_files)} PDFs in {data_path}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    success_count = 0
    fail_count = 0

    for i, pdf_file in enumerate(pdf_files):
        if i < start_index:
            continue

        logger.info(f"[{i+1}/{len(pdf_files)}] Processing {pdf_file.name}...")
        
        try:
            # Extract
            text = utils_p4.extract_text_from_pdf(pdf_file, language=language)
            text = sanitize_text(text)
            
            if not text or len(text.strip()) == 0:
                logger.warning(f"  Empty text for {pdf_file.name}. Skipping.")
                continue

            # Chunk
            doc_chunks = text_splitter.split_text(text)
            if not doc_chunks:
                logger.warning(f"  No chunks created for {pdf_file.name}. Skipping.")
                continue
                
            documents = [
                Document(page_content=chunk, metadata={"source": pdf_file.name, "language": language})
                for chunk in doc_chunks
            ]
            
            # Index (Add to Chroma)
            vector_store.add_documents(documents)
            logger.info(f"  Successfully indexed {len(documents)} chunks.")
            success_count += 1

        except Exception as e:
            logger.error(f"  FAILED to process {pdf_file.name}: {e}")
            fail_count += 1

    logger.info(f"--- Robust Indexing Complete ---")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0, help="Index of PDF to start from (0-based)")
    args = parser.parse_args()
    
    index_tamil_robust(start_index=args.start_index)

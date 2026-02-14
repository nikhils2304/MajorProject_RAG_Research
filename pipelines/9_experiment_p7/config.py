
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = BASE_DIR / "Datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
PIPELINE_DIR = BASE_DIR / "pipelines" / "9_experiment_p7"

# ChromaDB
CHROMA_DB_DIR = PIPELINE_DIR / "chroma_db_english_pivot"
COLLECTION_NAME = "english_pivot_collection"

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "gemini-2.0-flash"

# Dataset Mapping (Same as P4, but we'll filter top 25)
LANG_TO_PDF_DIR = {
    "English": "EnglishDataset",
    "Hindi": "HindiDataset",
    "Tamil": "TamilDataset",
    "Bengali": "BengaliDataset"
}

QA_DATASET_DIR = BASE_DIR / "pipelines" / "2_eval_dataset"

LANG_TO_QA_FILE = {
    "English": "english_qa.json",
    "Hindi": "hindi_qa.json",
    "Tamil": "tamil_qa.json",
    "Bengali": "bengali_qa.json"
}

# Translation Settings
TRANSLATION_TARGET = "en"
TRANSLATOR_ENGINE = "google" # deep_translator engine

# Constraints
DOCS_PER_LANG = 25

# Cost Settings
GEMINI_INPUT_PRICE_PER_1M = 0.10
GEMINI_OUTPUT_PRICE_PER_1M = 0.40


import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = BASE_DIR / "Datasets"
QA_DATASET_DIR = BASE_DIR / "pipelines" / "2_eval_dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
PIPELINE_DIR = BASE_DIR / "pipelines" / "10_experiment_p8"

# Reuse P7 Paths
P7_DIR = BASE_DIR / "pipelines" / "9_experiment_p7"
CHROMA_DB_DIR = P7_DIR / "chroma_db_english_pivot"
COLLECTION_NAME = "english_pivot_collection"

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "gemini-2.0-flash"

# Dataset Mapping
LANG_TO_PDF_DIR = {
    "English": "EnglishDataset",
    "Hindi": "HindiDataset",
    "Tamil": "TamilDataset",
    "Bengali": "BengaliDataset"
}

LANG_TO_QA_FILE = {
    "English": "english_qa.json",
    "Hindi": "hindi_qa.json",
    "Tamil": "tamil_qa.json",
    "Bengali": "bengali_qa.json"
}

# Translation Settings
TRANSLATION_TARGET = "en"
TRANSLATOR_ENGINE = "google"

# Hybrid Search Settings
TOP_K_DENSE = 60
TOP_K_SPARSE = 60
RRF_K = 60
FINAL_TOP_K = 5

# Cost Settings
GEMINI_INPUT_PRICE_PER_1M = 0.10
GEMINI_OUTPUT_PRICE_PER_1M = 0.40

# Reuse P7 translated docs directory
TRANSLATED_DOCS_DIR = P7_DIR / "translated_docs"

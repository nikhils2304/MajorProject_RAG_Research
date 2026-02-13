
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = BASE_DIR / "Datasets"
QA_DATASET_DIR = BASE_DIR / "pipelines/2_eval_dataset"
OUTPUT_DIR = BASE_DIR / "outputs" 

# ============================================================
# Model Config for P6 (Multilingual Reranked)
# ============================================================
MODEL_LABEL = "BAAI/bge-m3 + BAAI/bge-reranker-v2-m3"
MODELS = {
    "English": "BAAI/bge-m3",
    "Hindi": "BAAI/bge-m3",
    "Tamil": "BAAI/bge-m3",
    "Bengali": "BAAI/bge-m3",
}
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Directory for Vector Stores (REUSE P4 SHARED INDEX)
CHROMA_DB_DIR = BASE_DIR / "pipelines/6_experiment_p4/chroma_db_multilingual" 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval Settings
TOP_K_RETRIEVAL = 50 # Retrieve more candidates
TOP_K_RERANK = 10    # Rerank to top 10

# Dataset Mapping
LANG_TO_FOLDER = {
    "English": "EnglishDataset",
    "Hindi": "HindiDataset",
    "Tamil": "TamilDataset",
    "Bengali": "BengaliDataset",
}

LANG_TO_QA_FILE = {
    "English": "english_qa.json",
    "Hindi": "hindi_qa.json",
    "Tamil": "tamil_qa.json",
    "Bengali": "bengali_qa.json",
}

# Cost Settings
GEMINI_INPUT_PRICE_PER_1M = 0.10
GEMINI_OUTPUT_PRICE_PER_1M = 0.40

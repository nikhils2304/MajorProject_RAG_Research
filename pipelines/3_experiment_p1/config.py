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
# Model Profiles for Comparison
# ============================================================
# Switch ACTIVE_MODEL_PROFILE to run experiments with different models.
# Each profile gets its own ChromaDB directory so indices don't clash.

ACTIVE_MODEL_PROFILE = "indicsbert"  # Options: "vyakyarth", "indicsbert"

MODEL_PROFILES = {
    "vyakyarth": {
        "label": "Vyakyarth-1",
        "models": {
            "English": "BAAI/bge-m3",
            "Hindi": "krutrim-ai-labs/Vyakyarth",
            "Tamil": "krutrim-ai-labs/Vyakyarth",
            "Bengali": "krutrim-ai-labs/Vyakyarth",
        },
        "chroma_dir": "chroma_db_vyakyarth",
    },
    "indicsbert": {
        "label": "L3Cube-IndicSBERT",
        "models": {
            "English": "BAAI/bge-m3",
            "Hindi": "l3cube-pune/indic-sentence-bert-nli",
            "Tamil": "l3cube-pune/indic-sentence-bert-nli",
            "Bengali": "l3cube-pune/indic-sentence-bert-nli",
        },
        "chroma_dir": "chroma_db_indicsbert",
    },
}

# Resolved config from active profile
_profile = MODEL_PROFILES[ACTIVE_MODEL_PROFILE]
MODELS = _profile["models"]
MODEL_LABEL = _profile["label"]
CHROMA_DB_DIR = BASE_DIR / "pipelines/3_experiment_p1" / _profile["chroma_dir"]

# Create directories if they don't exist
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval Settings
TOP_K = 10  # Retrieve top 10 docs (for Recall@10)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

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

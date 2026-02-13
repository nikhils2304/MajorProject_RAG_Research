
import sys
from pathlib import Path
import logging

# Add P1 directory to path to reuse utilities
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))

# Import from P1
# NOTE: importing utils_p1 triggers basicConfig writing to experiment_p1.log
from utils_p1 import extract_text_from_pdf, get_pdf_files

# Reset logging handlers to force writing to experiment_p4.log
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_p4.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

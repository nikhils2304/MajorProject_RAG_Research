import pandas as pd
from pathlib import Path
import sys

# Define file paths
OUTPUT_DIR = Path("outputs")

# P1 Files (IndicSBERT for Indic, BAAI/bge-m3 for English)
P1_FILES = {
    "English": "experiment_p1_english_results_20260211_145123.csv",
    "Hindi": "experiment_p1_hindi_indicsbert_results_20260212_131645.csv",
    "Tamil": "experiment_p1_tamil_indicsbert_results_20260212_132059.csv",
    "Bengali": "experiment_p1_bengali_indicsbert_results_20260212_132504.csv"
}

# P2 Files (Hybrid)
P2_FILES = {
    "English": "experiment_p2_english_results_20260212_135921.csv",
    "Hindi": "experiment_p2_hindi_results_20260212_140529.csv",
    "Tamil": "experiment_p2_tamil_results_20260212_141006.csv",
    "Bengali": "experiment_p2_bengali_results_20260212_141421.csv"
}

# P3 Files (Reranked)
P3_FILES = {
    "English": "experiment_p3_english_results_20260212_150052.csv",
    "Hindi": "experiment_p3_hindi_results_20260212_151219.csv",
    "Tamil": "experiment_p3_tamil_results_20260212_151946.csv",
    "Bengali": "experiment_p3_bengali_results_20260212_152602.csv"
}

def load_metrics(filepath, language, experiment):
    path = OUTPUT_DIR / filepath
    if not path.exists():
        print(f"Warning: File not found {path}")
        return None
    
    df = pd.read_csv(path)
    return {
        "Language": language,
        "Experiment": experiment,
        "Recall@5": df["hit_rate@5"].mean(),
        "Recall@10": df["hit_rate@10"].mean(),
        "BLEU": df["bleu"].mean(),
        "ROUGE-L": df["rouge_l"].mean()
    }

results = []

# Load P1
for lang, f in P1_FILES.items():
    res = load_metrics(f, lang, "P1 (Dense)")
    if res: results.append(res)

# Load P2
for lang, f in P2_FILES.items():
    res = load_metrics(f, lang, "P2 (Hybrid)")
    if res: results.append(res)

# Load P3
for lang, f in P3_FILES.items():
    res = load_metrics(f, lang, "P3 (Reranked)")
    if res: results.append(res)

# Create DataFrame
df = pd.DataFrame(results)

# Sort by Language and Experiment
df = df.sort_values(by=["Language", "Experiment"])

# Print Markdown Table
print("\n### Final Comparison: P1 vs P2 vs P3\n")
print(df.to_markdown(index=False, floatfmt=".4f"))

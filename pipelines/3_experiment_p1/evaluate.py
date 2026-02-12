import sys
from pathlib import Path
import json
import logging
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
import google.generativeai as genai
import os
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent))

import config
import utils_p1
from embeddings import MonolingualEmbedding

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

logger = utils_p1.logger

def generate_answer(query: str, context_docs: list) -> str:
    """Generates an answer using Gemini given retrieved context."""
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant. Answer the user's question based strictly on the provided context.
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer."

def evaluate_language(language: str):
    """
    Runs the full RAG evaluation loop for a given language.
    """
    logger.info(f"--- Starting Evaluation for {language} (Model: {config.MODEL_LABEL}) ---")
    
    # 1. Load QA Dataset
    qa_file = config.QA_DATASET_DIR / config.LANG_TO_QA_FILE[language]
    if not qa_file.exists():
        logger.error(f"QA file not found: {qa_file}")
        return

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_file}")

    # 2. Initialize Vector Store (Read-Only)
    persist_dir = config.CHROMA_DB_DIR / f"{language.lower()}_index"
    model_name = config.MODELS[language]
    embedding_fn = MonolingualEmbedding(model_name=model_name)
    
    vector_store = Chroma(
        collection_name=f"{language.lower()}_collection",
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir)
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K})

    # 3. Evaluation Loop
    results = []
    
    for item in tqdm(qa_pairs, desc=f"Evaluating {language}"):
        query = item.get("question")
        gold_answer = item.get("answer")
        gold_source = item.get("source_file") # Filename of the source PDF
        
        # Retrieval
        retrieved_docs = retriever.invoke(query)
        
        # Metrics Calculation
        # Hit Rate @ K (Did we retrieve the correct document?)
        hits = [1 if gold_source in doc.metadata.get("source", "") else 0 for doc in retrieved_docs]
        hit_rate_at_5 = 1 if sum(hits[:5]) > 0 else 0
        hit_rate_at_10 = 1 if sum(hits[:10]) > 0 else 0
        
        # Generation & Metrics
        generated_answer = generate_answer(query, retrieved_docs[:5])
        
        # BLEU Score
        bleu = BLEU()
        bleu_score = bleu.sentence_score(generated_answer, [gold_answer]).score
        
        # ROUGE-L Score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, generated_answer)
        rouge_l = rouge_scores['rougeL'].fmeasure

        results.append({
            "language": language,
            "question": query,
            "gold_source": gold_source,
            "hit_rate@5": hit_rate_at_5,
            "hit_rate@10": hit_rate_at_10,
            "bleu": bleu_score,
            "rouge_l": rouge_l,
            "generated_answer": generated_answer,
            "gold_answer": gold_answer
        })

    # 4. Aggregate Results
    df = pd.DataFrame(results)
    avg_hit_5 = df["hit_rate@5"].mean()
    avg_hit_10 = df["hit_rate@10"].mean()
    avg_bleu = df["bleu"].mean()
    avg_rouge = df["rouge_l"].mean()
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    profile_tag = config.ACTIVE_MODEL_PROFILE
    output_path = config.OUTPUT_DIR / f"experiment_p1_{language.lower()}_{profile_tag}_results_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Evaluation Complete for {language}")
    logger.info(f"Hit Rate @ 5: {avg_hit_5:.4f}")
    logger.info(f"Hit Rate @ 10: {avg_hit_10:.4f}")
    logger.info(f"Avg BLEU: {avg_bleu:.4f}")
    logger.info(f"Avg ROUGE-L: {avg_rouge:.4f}")
    logger.info(f"Detailed results saved to {output_path}")
    
    return {
        "Language": language,
        "Model": config.MODEL_LABEL,
        "Recall@5": avg_hit_5,
        "Recall@10": avg_hit_10,
        "BLEU": avg_bleu,
        "ROUGE-L": avg_rouge
    }

if __name__ == "__main__":
    final_metrics = []
    # Evaluate Indic languages with the active model profile
    for lang in ["Hindi", "Tamil", "Bengali"]:
        metrics = evaluate_language(lang)
        if metrics:
            final_metrics.append(metrics)
            
    # Print Summary Table
    summary_df = pd.DataFrame(final_metrics)
    print(f"\n--- Experiment P1 Summary (Model: {config.MODEL_LABEL}) ---")
    print(summary_df.to_markdown(index=False))

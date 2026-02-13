
import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
import google.generativeai as genai
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import os
import time

# Add P6 directory
sys.path.append(str(Path(__file__).parent))
import config
import utils_p6

# Add P1 directory for embeddings
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))
from embeddings import MonolingualEmbedding

# Import P6 Reranker
from reranker import Reranker

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

logger = utils_p6.logger

def calculate_faithfulness(question, answer, context_text):
    """LLM-as-Judge for Faithfulness"""
    try:
        prompt = f"""You are an unbiased evaluator.
        Context: {context_text}
        Question: {question}
        Answer: {answer}
        Is the Answer faithful to the Context? respond with strictly "YES" or "NO"."""
        response = model.generate_content(prompt)
        return 1.0 if "YES" in response.text.strip().upper() else 0.0
    except:
        return 0.0
        
def generate_answer_with_cost(query, context_docs):
    """Generates answer and tracks cost."""
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant. Answer the user's question based strictly on the provided context.
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        cost = ((input_tokens / 1e6) * config.GEMINI_INPUT_PRICE_PER_1M) + \
               ((output_tokens / 1e6) * config.GEMINI_OUTPUT_PRICE_PER_1M)
               
        return text, cost, context_text
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error", 0.0, ""

def evaluate_language(language: str):
    logger.info(f"--- Starting P6 Reranked Evaluation for {language} ---")
    
    # 1. Load QA Data
    qa_file = config.QA_DATASET_DIR / config.LANG_TO_QA_FILE[language]
    if not qa_file.exists():
        logger.error(f"QA file not found: {qa_file}")
        return None
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    # 2. Initialize Shared Vector Store
    persist_dir = config.CHROMA_DB_DIR / "shared_index"
    embedding_fn = MonolingualEmbedding(model_name=config.MODELS["English"])
    
    vector_store = Chroma(
        collection_name="shared_multilingual_collection",
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir)
    )
    
    # 3. Initialize Reranker
    reranker = Reranker(model_name=config.RERANKER_MODEL)

    # 4. Evaluation Loop
    results = []
    total_cost = 0.0
    
    for item in tqdm(qa_pairs, desc=f"Evaluating {language}"):
        query = item.get("question")
        gold_answer = item.get("answer")
        gold_source = item.get("source_file")
        
        # Step 1: Dense Retrieval (Top-50 candidates)
        candidates = vector_store.similarity_search(query, k=config.TOP_K_RETRIEVAL)
        
        # Step 2: Reranking (Top-10 final)
        if candidates:
            final_docs = reranker.rerank(query, candidates, top_k=config.TOP_K_RERANK)
        else:
            final_docs = []
            
        # Retrieval Metrics (on Final Docs)
        hits = [1 if gold_source in doc.metadata.get("source", "") else 0 for doc in final_docs]
        hit_rate_at_5 = 1 if sum(hits[:5]) > 0 else 0
        hit_rate_at_10 = 1 if sum(hits[:10]) > 0 else 0
        
        # Generation
        generated_answer, cost, context_text = generate_answer_with_cost(query, final_docs[:5])
        total_cost += cost
        
        # Metrics
        bleu = BLEU()
        bleu_score = bleu.sentence_score(generated_answer, [gold_answer]).score
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(gold_answer, generated_answer)['rougeL'].fmeasure
        
        faithfulness = calculate_faithfulness(query, generated_answer, context_text)
        
        results.append({
            "language": language,
            "hit_rate@5": hit_rate_at_5,
            "hit_rate@10": hit_rate_at_10,
            "bleu": bleu_score,
            "rouge_l": rouge_l,
            "faithfulness": faithfulness,
            "cost": cost
        })

    # 5. Aggregate
    df = pd.DataFrame(results)
    avg_re_5 = df["hit_rate@5"].mean()
    avg_re_10 = df["hit_rate@10"].mean()
    avg_bleu = df["bleu"].mean()
    avg_rouge = df["rouge_l"].mean()
    avg_faith = df["faithfulness"].mean()
    cost_per_1k = (df["cost"].sum() / len(df)) * 1000 if len(df) > 0 else 0.0

    logger.info(f"P6 Evaluation Complete for {language}")
    logger.info(f"Recall@10: {avg_re_10:.4f}")
    
    return {
        "Language": language,
        "Recall@5": avg_re_5,
        "Recall@10": avg_re_10,
        "BLEU": avg_bleu,
        "ROUGE-L": avg_rouge,
        "Faithfulness": avg_faith,
        "Cost_per_1k": cost_per_1k
    }

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str)
    args = parser.parse_args()
    
    languages = [args.language] if args.language else ["English", "Hindi", "Tamil", "Bengali"]
    final_metrics = []
    
    for lang in languages:
        m = evaluate_language(lang)
        if m: final_metrics.append(m)
        
    summary_df = pd.DataFrame(final_metrics)
    print("\n--- Experiment P6 Summary (Multilingual Reranked) ---")
    print(summary_df.to_markdown(index=False))

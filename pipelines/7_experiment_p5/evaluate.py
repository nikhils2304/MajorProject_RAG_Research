
import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain_core.documents import Document
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import os
import time

# Add P5 directory
sys.path.append(str(Path(__file__).parent))
import config
import utils_p5

# Add P1 directory for embeddings
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))
from embeddings import MonolingualEmbedding

# Import P5 modules (copied from P2)
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

logger = utils_p5.logger

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
    logger.info(f"--- Starting P5 Hybrid Evaluation for {language} ---")
    
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
    
    # 3. Build BM25 Index (Fetch all docs from Chroma)
    logger.info("Fetching all documents from Shared Index for BM25...")
    collection = vector_store._collection
    all_data = collection.get(include=["documents", "metadatas"])
    
    all_docs = []
    if all_data["documents"]:
        for i, doc_text in enumerate(all_data["documents"]):
            metadata = all_data["metadatas"][i] if all_data["metadatas"] else {}
            all_docs.append(Document(page_content=doc_text, metadata=metadata))
            
    logger.info(f"Loaded {len(all_docs)} docs. Building BM25 index...")
    bm25_retriever = BM25Retriever(all_docs)
    
    # 4. Initialize Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        rrf_k=config.RRF_K
    )

    # 5. Evaluation Loop
    results = []
    total_cost = 0.0
    
    for item in tqdm(qa_pairs, desc=f"Evaluating {language}"):
        query = item.get("question")
        gold_answer = item.get("answer")
        gold_source = item.get("source_file")
        
        # Retrieval
        retrieved_docs = hybrid_retriever.retrieve(query, top_k=config.TOP_K)
        
        # Hit Rate
        hits = [1 if gold_source in doc.metadata.get("source", "") else 0 for doc in retrieved_docs]
        hit_rate_at_5 = 1 if sum(hits[:5]) > 0 else 0
        hit_rate_at_10 = 1 if sum(hits[:10]) > 0 else 0
        
        # Generation
        generated_answer, cost, context_text = generate_answer_with_cost(query, retrieved_docs[:5])
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

    # 6. Aggregate
    df = pd.DataFrame(results)
    avg_re_5 = df["hit_rate@5"].mean()
    avg_re_10 = df["hit_rate@10"].mean()
    avg_bleu = df["bleu"].mean()
    avg_rouge = df["rouge_l"].mean()
    avg_faith = df["faithfulness"].mean()
    cost_per_1k = (df["cost"].sum() / len(df)) * 1000 if len(df) > 0 else 0.0

    logger.info(f"P5 Evaluation Complete for {language}")
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
    print("\n--- Experiment P5 Summary (Multilingual Hybrid) ---")
    print(summary_df.to_markdown(index=False))

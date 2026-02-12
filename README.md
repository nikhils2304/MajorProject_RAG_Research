# Multilingual RAG Research — Project README

## 📌 Project Overview

This project investigates **Retrieval-Augmented Generation (RAG)** performance across multiple Indian languages by systematically evaluating different embedding and vector search strategies.

The research follows a structured experimental design to understand how retrieval quality and generation quality vary across languages and retrieval configurations.

The study begins with **mono-lingual RAG evaluation** and then expands into **multi-lingual RAG systems**.

---

# 🎯 Research Objectives

The primary objectives are:

* Evaluate embedding strategies for Indian languages
* Compare vector search strategies (dense, hybrid, reranked)
* Measure retrieval quality and answer quality
* Analyze performance differences across languages
* Measure latency and efficiency for real-world deployment

All experiments are conducted under controlled conditions:

> Only embedding strategy and retrieval strategy change.
> All other variables remain constant.

---

# 🌏 Selected Languages

The research will be conducted on the following languages:

| Language    | Purpose                                        |
| ----------- | ---------------------------------------------- |
| **English** | Baseline, dataset creation, pipeline debugging |
| **Hindi**   | High-resource Indian language                  |
| **Tamil**   | Dravidian language representation              |
| **Bengali** | Indo-Aryan language with different script      |

These languages provide:

* Strong corpus availability
* Good LLM support
* Script diversity
* Reliable QA evaluation capability

---

# 🧪 Experiment Design

## Experiment Definitions

| Experiment_ID | Embedding_Strategy | Search_Strategy | Description |
| :--- | :--- | :--- | :--- |
| **P1** | Monolingual | Dense | Mono-lingual dense retrieval |
| **P2** | Monolingual | Hybrid | Mono-lingual hybrid retrieval |
| **P3** | Monolingual | Reranked | Mono-lingual reranked |
| **P4** | Multilingual | Dense | Shared-space dense |
| **P5** | Multilingual | Hybrid | Shared-space hybrid |
| **P6** | Multilingual | Reranked | Shared-space reranked |
| **P7** | Translate→Embed | Dense | Pivot-based dense |
| **P8** | Translate→Embed | Hybrid | Pivot-based hybrid |
| **P9** | Translate→Embed | Reranked | Pivot-based reranked |

### Phase 1 — Mono-lingual Experiments

> **Values**: P1, P2, P3  
> **Starting point**: P1: English (Monolingual + Dense) — extends to Hindi, Tamil, Bengali. This establishes the baseline RAG pipeline.

### Phase 2 — Multi-lingual Experiments (Later Stage)

> **Values**: P4, P5, P6 (Shared Space), P7, P8, P9 (Translation Pivot)

---

# 📚 Dataset Creation Strategy

Each language requires a high-quality QA evaluation dataset.

## Current Notebook Capability

A working notebook already exists that:

* Ingests PDF documents
* Generates QA pairs using Gemini
* Critiques QA using LLM-as-Judge
* Filters high-quality QA pairs
* Builds vector index
* Runs RAG pipeline
* Computes evaluation metrics

Currently tested for:

> English QA generation and evaluation

---

## QA Dataset Protocol (English P1)

### Source

PDF-based document corpus

### QA generation target

Generate:

> 300–400 QA pairs

Filter to:

> 200–250 high-quality QA pairs

### Filtering Criteria

Each QA pair scored (1–5):

* Groundedness
* Relevance
* Standalone clarity
* Answer completeness

Keep only:

> All scores ≥ 4

The filtered dataset becomes:

> **English Evaluation Dataset v1.0 (Frozen)**

This dataset will be reused across:

* P1
* P2
* P3

---

# ⚙️ RAG Pipeline Structure

Each experiment follows:

1. Extract text from PDFs
2. Chunk documents
3. Generate embeddings
4. Store in vector database
5. Retrieve top-k contexts
6. Generate answer using LLM
7. Compute evaluation metrics

---

# 📊 Evaluation Metrics (Finalized)

These are the official research metrics:

| Metric              | Purpose                     |
| ------------------- | --------------------------- |
| **Recall@5**        | Strict retrieval accuracy   |
| **Recall@10**       | Expanded retrieval accuracy |
| **BLEU**            | Answer similarity           |
| **Average Latency** | Efficiency                  |

Optional diagnostics:

* MRR
* Faithfulness (later stage)

---

# 📈 Final Research Tables

## Table 1 — Mono-lingual Performance (P1 - Dense)

| Language | Experiment_ID | Recall@5 | Recall@10 | Bleu | Avg_Latency_ms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EN** | P1 | 0.9786 | | 33.12 | |
| **HI** | P1 | 0.8130 | | 27.95 | |
| **TA** | P1 | 0.6099 | 0.7309 | 7.57 | |
| **BG** | P1 | 0.7356 | 0.8173 | 11.93 | |

## Table 2 — Mono-lingual Performance (P2 - Hybrid)

| Language | Experiment_ID | Recall@5 | Recall@10 | Bleu | Avg_Latency_ms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EN** | P2 | | | | |
| **HI** | P2 | | | | |
| **TA** | P2 | | | | |
| **BG** | P2 | | | | |

## Table 3 — Mono-lingual Performance (P3 - Reranked)

| Language | Experiment_ID | Recall@5 | Recall@10 | Bleu | Avg_Latency_ms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EN** | P3 | | | | |
| **HI** | P3 | | | | |
| **TA** | P3 | | | | |
| **BG** | P3 | | | | |

## Table 4 — Multi-lingual Experiment Results

| Experiment_ID | Recall@5 | Recall@10 | Bleu | Faithfulness | Latency_ms | Cost_per_1k_queries |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P4** | | | | | | |
| **P5** | | | | | | |
| **P6** | | | | | | |
| **P7** | | | | | | |
| **P8** | | | | | | |
| **P9** | | | | | | |

## Table 5 — Cross-Lingual Recall Breakdown

| Query_Language | Document_Language | Experiment_ID | Recall@10 |
| :--- | :--- | :--- | :--- |
| **HI** | EN | P4 | |
| **TA** | EN | P4 | |
| **ML** | EN | P4 | |
| **BN** | EN | P4 | |
| **HI** | EN | P7 | |
| **TA** | EN | P7 | |

---

# 🚀 Starting Experiment: P1 English

### Goal

Establish baseline RAG performance using:

* English dataset
* Monolingual embeddings
* Dense retrieval

### Expected Output

| Language | Experiment | Recall@5 | Recall@10 | BLEU | Latency |
| -------- | ---------- | -------- | --------- | ---- | ------- |
| EN       | P1         | X        | X         | X    | X       |

This becomes the baseline for the entire study.

---

# 🗂 Research Workflow

## Phase 1

Generate and freeze English dataset

## Phase 2

Run P1 evaluation

## Phase 3

Run P2 and P3

## Phase 4

Repeat for Hindi, Tamil, Bengali

## Phase 5

Run multilingual experiments

## Phase 6

Fill final research tables and analysis

---

# ⚠️ Experimental Constraints

* Dataset must remain frozen after filtering
* Same dataset used across P1–P3
* Same LLM across experiments
* Same chunking and embedding parameters
* Only retrieval strategy changes per experiment

---

# 📍 Current Status

✔️ Languages finalized
✔️ Metrics finalized
✔️ Table structure finalized
✔️ QA generation pipeline fixed for Tamil/Bengali
✔️ P1 Experiments complete for all 4 languages

### Next Steps

1. Review P1 results and investigate potential for Hybrid search (P2)
2. Begin multi-lingual experiment planning (P4)

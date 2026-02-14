# Multilingual RAG Research Report

## Experiments Overview

| Experiment | Strategy | Model(s) |
| :--- | :--- | :--- |
| **P1** | **Dense Retrieval** | **English:** `BAAI/bge-m3`<br>**Indic:** `l3cube-pune/indic-sentence-bert-nli` (IndicSBERT) |
| **P2** | **Hybrid Search** | Dense Retrieval (as above) + **BM25** (Sparse)<br>Fused via **Reciprocal Rank Fusion (RRF)** |
| **P3** | **Reranked Search** | Dense Retrieval (as above, top-20) + **Cross-Encoder Reranking**<br>Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |

## Results Comparison

Below is the consolidated performance table across all experiments.

| Language | Experiment | Recall@5 | Recall@10 | BLEU | ROUGE-L |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bengali** | P1 (Dense) | 0.9038 | 0.9231 | 21.79 | 0.0150 |
| | **P2 (Hybrid)** | **0.9519** | **0.9712** | **26.37** | **0.0243** |
| | P3 (Reranked) | 0.9327 | 0.9567 | 25.08 | 0.0251 |
| **English** | P1 (Dense) | 0.9786 | 0.9829 | 33.12 | 0.5587 |
| | P2 (Hybrid) | 0.9744 | 0.9829 | 32.79 | 0.5574 |
| | **P3 (Reranked)** | **0.9786** | **0.9829** | **33.99** | **0.5789** |
| **Hindi** | P1 (Dense) | 0.9024 | 0.9472 | 31.35 | 0.3814 |
| | **P2 (Hybrid)** | **0.9472** | **0.9634** | **38.04** | 0.4142 |
| | P3 (Reranked) | 0.9472 | 0.9512 | 36.88 | **0.4192** |
| **Tamil** | P1 (Dense) | 0.7668 | 0.8700 | 13.56 | 0.1149 |
| | **P2 (Hybrid)** | **0.8610** | **0.8969** | 15.73 | 0.1186 |
| | P3 (Reranked) | 0.8475 | **0.8969** | **16.14** | **0.1531** |

### Key Findings

1.  **Hybrid Search (P2) Wins for Indic Languages**:
    *   Combining keyword search (BM25) with semantic search (Dense) consistently yielded the highest recall for Bengali, Hindi, and Tamil. This suggests that dense embeddings alone might miss exact keyword matches in these morphologically rich languages.
2.  **Reranking (P3) Improves Precision**:
    *   While its recall was limited by the initial candidate pool (we used top-20 for speed), the reranker effectively ordered relevant documents to the top, improving generation metrics (BLEU/ROUGE) in several cases (especially English and Tamil).
3.  **English is Robust**:
    *   Performance for English was high (>97% recall) across all methods, with reranking providing a slight edge in answer quality.

## Next Steps

*   **Fine-tuning**: Consider fine-tuning the embedding model on domain-specific data to improve dense retrieval.
*   **Larger Candidate Pool**: For P3, increasing `CANDIDATE_K` (e.g., to 50 or 100) on a GPU-enabled machine would likely improve recall further.

## Latency Analysis

Average end-to-end latency (retrieval + generation) per query in milliseconds:

| Language | P1 (Dense) | P2 (Hybrid) | P3 (Reranked) |
| :--- | :--- | :--- | :--- |
| **Bengali** | 1426 | **1223** | 1809 |
| **English** | 1594 | **1360** | 3494 |
| **Hindi** | 1477 | 1497 | 2793 |
| **Tamil** | 1548 | **1243** | 2004 |

*   **P2 (Hybrid) is the fastest**: The overhead of BM25 is negligible, and it seems to retrieve faster or lead to faster generation (possibly due to shorter/better context?).
*   **P3 (Reranked) adds ~500ms - 2000ms**: The cross-encoder reranking step adds significant latency, especially for English and Hindi in this run.

*   **P3 (Reranked) adds ~500ms - 2000ms**: The cross-encoder reranking step adds significant latency, especially for English and Hindi in this run.

## P4: Multilingual Dense Retrieval (Shared-Space)

**Status:** Complete.

**Strategy:** Shared Multilingual Index (`BAAI/bge-m3`) tested with monolingual queries.

| Language | Experiment | Recall@5 | Recall@10 | BLEU | ROUGE-L | Faithfulness | Cost/1k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| **Bengali** | P4 (Multi) | 0.9519 | **0.9615** | 28.13 | 0.0257 | 0.9471 | $0.0930 |
| **English** | P4 (Multi) | 0.9744 | 0.9786 | 33.72 | **0.5647** | 0.9615 | $0.0887 |
| **Hindi** | P4 (Multi) | 0.9797 | **0.9959** | **40.62** | 0.4490 | 0.9837 | $0.1188 |
| **Tamil** | P4 (Multi) | 0.8341 | 0.9283 | 15.46 | 0.1411 | 0.7937 | $0.1220 |

### Key Findings (P4)
1.  **High Recall in Shared Space**: Despite sharing the vector space with 4 languages (~10k docs), `BAAI/bge-m3` maintained extremely high recall for Hindi (99.6%) and English (97.9%).
2.  **Robustness**: Tamil recall reached 92.8% even with the indexing challenges.
3.  **Cost Effective**: ~$0.10 per 1k queries.

To run P4:

1.  **Index/Eval**: `python pipelines/6_experiment_p4/evaluate.py`

## P5: Multilingual Hybrid Retrieval (Reciprocal Rank Fusion)

**Status:** Complete.

**Strategy:** Shared Index (Dense) + In-Memory BM25 (Sparse) fused via RRF (k=60).

| Language | Experiment | Recall@5 | Recall@10 | BLEU | ROUGE-L | Faithfulness | Cost/1k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| **Bengali** | P5 (Hybrid) | **0.9615** | **0.9712** | 27.97 | **0.0308** | 0.8942 | $0.0913 |
| **English** | P5 (Hybrid) | 0.9744 | **0.9829** | 32.48 | 0.5635 | 0.9274 | $0.0875 |
| **Hindi** | P5 (Hybrid) | **0.9797** | **1.0000** | **42.03** | 0.4445 | 0.9756 | $0.1184 |
| **Tamil** | P5 (Hybrid) | 0.7758 | 0.8879 | **17.62** | 0.1297 | 0.7130 | $0.1272 |

### Key Findings (P5)
1.  **Hindi Achieves Perfection**: Hybrid search pushed Hindi Recall@10 to **100%** (246/246 queries found correct doc). BM25 likely fixed the few cases Dense missed.
2.  **Trade-off for Tamil**: Tamil Recall dropped (92.8% -> 88.8%). This suggests the simple BM25 tokenizer struggles with Tamil morphology, and RRF weighted the poor sparse results too heavily against the good dense results.
3.  **BLEU Improvement**: Aggregate BLEU (30.0) is slightly higher than P4 (29.5), indicating better precision in top ranks.


## P6: Multilingual Reranked Retrieval

**Status:** Paused (Hardware Constraints).

**Strategy:** Dense Retrieval (Top-20) + Cross-Encoder Reranking (`BAAI/bge-reranker-v2-m3`).

**Outcome:**
The SOTA reranker model (560M params) was too computationally expensive for local execution (~2 mins per query).
The experiment setup is complete and can be resumed on a machine with a powerful GPU (e.g., A100/H100 or M1 Max/Ultra).

To resume:
`python pipelines/8_experiment_p6/evaluate.py`

## Project Repository

The complete project code, including datasets and ChromaDB indices, has been uploaded to:
[GitHub Repository](https://github.com/nikhils2304/MajorProject_RAG_Research)

## P7: Pivot-Based Dense Retrieval (Translate -> Embed -> Retrieve -> Translate Back)

**Status:** Complete.

**Strategy:** 
1.  **Translation**: All Documents (Indic) -> English. Queries (Indic) -> English.
2.  **Indexing**: `BAAI/bge-m3` on English Corpus only.
3.  **Retrieval**: Dense Search in English.
4.  **Generation**: Answer in English -> Translate back to Indic.

| Language | Recall@5 | Recall@10 | BLEU | ROUGE-L | Faithfulness | Latency | Cost/1k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bengali** | 0.9588 | 0.9691 | 8.82 | 0.0165 | 0.9691 | 3548ms | $0.127 |
| **English** | 1.0000 | 1.0000 | 36.49 | 0.6123 | 0.9606 | 1549ms | $0.161 |
| **Hindi** | 0.9920 | 1.0000 | 29.23 | 0.3698 | 1.0000 | 3903ms | $0.118 |
| **Tamil** | 0.9750 | 0.9917 | 13.11 | 0.1632 | 0.9833 | 3483ms | $0.118 |

### Key Findings (P7)
1.  **100% Recall for English/Hindi**: The English-centric retrieval worked perfectly for Hindi queries translated to English (Recall@10 = 1.0).
2.  **Latency Penalty**: The double translation (Query -> English, Answer -> Indic) adds significant latency (~2s overhead compared to English).
3.  **Low ROUGE for Indic**: Despite high retrieval accuracy and good Faithfulness (>0.96), ROUGE scores are low for Bengali and Tamil. This indicates that **back-translation** produces phrasing that differs significantly from the ground truth reference, even if the meaning is correct.

## P8: Pivot-Based Hybrid Retrieval (Translate -> Dense+BM25 -> RRF -> Translate Back)

**Status:** Complete.

**Strategy:** 
1.  **Translation**: Query (Indic) -> English.
2.  **Hybrid Retrieval**: English Dense (`BAAI/bge-m3`) + English Sparse (`BM25`) -> Fused via RRF.
3.  **Generation**: Answer in English.
4.  **Translation**: Answer (English) -> Indic.

| Language | Recall@5 | Recall@10 | BLEU | ROUGE-L | Faithfulness | Latency | Cost/1k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bengali** | 0.8763 | 0.9485 | 6.96 | 0.0098 | 0.8866 | 4771ms | $0.125 |
| **English** | 0.8819 | 0.9921 | 25.28 | 0.4490 | 0.9055 | 1487ms | $0.194 |
| **Hindi** | 0.9680 | 0.9920 | 24.33 | 0.3413 | 0.9520 | 4754ms | $0.120 |
| **Tamil** | 0.9917 | 1.0000 | 11.71 | 0.1257 | 0.9667 | 4834ms | $0.120 |
| **Aggregate** | **0.9318** | **0.9851** | **17.77** | **0.2468** | **0.9296** | **3893ms** | **$0.141** |

### Key Findings (P8)
1.  **Recall Dip**: Surprisingly, adding BM25 via RRF *lowered* Recall@5 for Bengali and English compared to Pure Dense (P7). This suggests BM25 rankings for translated terms might be noisy, diluting the strong dense signal in the top ranks.
2.  **High Latency**: The combination of Translation + Hybrid Search + RRF + Back-Translation makes this the slowest pipeline (~4s/query for Indic).
3.  **Tamil & Hindi Robustness**: Tamil hit 100% Recall@10, showing that for some languages, keyword signals helper recover documents that dense retrieval missed.


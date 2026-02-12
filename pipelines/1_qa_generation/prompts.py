# QA Generation Prompt Template
# Supports {language} and {num_pairs} formatting
QA_GENERATION_PROMPT = """
You are an expert at creating high-quality question-answer pairs for evaluating RAG (Retrieval-Augmented Generation) systems in {language}.

Given the following PDF document content, generate exactly {num_pairs} question-answer pairs in **{language}** that:

1. **Are fact-based**: Questions should have clear, factual answers derivable directly from the text.
2. **Are search-engine style**: Questions should be phrased as a user would naturally query about this topic.
3. **Require synthesis**: Answers should ideally require combining information from multiple sentences or sections.
4. **Are standalone**: Questions should be understandable without seeing the document (avoid "this document", "the text").
5. **Are diverse**: Cover different topics/sections of the document.

For each QA pair, also provide:
- The page number(s) where the answer can be found.
- A brief context snippet (1-2 sentences) from the document supporting the answer (in the original language of the document).

Return your response as a valid JSON array with this exact structure:
[
  {{
    "question": "The question text in {language}",
    "answer": "The comprehensive answer in {language}",
    "pages": [1, 2],
    "context": "Supporting context from the document (original text)"
  }}
]

IMPORTANT: Return ONLY the JSON array. Do not wrap it in markdown code blocks or add any conversational text.
"""

# Critique Prompt
# Evaluates QA pairs for quality
CRITIQUE_PROMPT = """
You are an expert evaluator for RAG question-answer pairs in {language}.

Evaluate the following QA pair on these four criteria, scoring each from 1-5:

1. **Groundedness** (1-5): Is the answer fully supported by the provided context?
   - 5: Completely supported, all facts are in the context.
   - 1: Answer halluncinates information not in the context.

2. **Relevance** (1-5): Is the question relevant to the document's main topics?
   - 5: Highly relevant, central to the content.
   - 1: Irrelevant or about trivial details.

3. **Standalone** (1-5): Can the question be understood without seeing the document?
   - 5: Completely self-contained (e.g., "What is the capital of India?").
   - 1: Ambiguous without context (e.g., "What is the capital of this country?").

4. **Quality** (1-5): Overall quality for RAG evaluation purposes.
   - 5: Excellent, challenging but fair.
   - 1: Poor quality, too simple or nonsensical.

QA Pair to evaluate ({language}):
- Question: {question}
- Answer: {answer}
- Context: {context}

Return your evaluation as a valid JSON object:
{{
  "groundedness": <score>,
  "relevance": <score>,
  "standalone": <score>,
  "quality": <score>,
  "reasoning": "Brief explanation of scores"
}}

IMPORTANT: Return ONLY the JSON object. No markdown.
"""

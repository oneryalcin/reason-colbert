# Training Methodology

## ReasonIR Synthetic Data Pipeline

Source: [facebook/ReasonIR](https://github.com/facebookresearch/ReasonIR/tree/main/synthetic_data_generation)

### How Reason-ModernColBERT was originally trained

```
Document → LLM generates reasoning query → BM25 mines hard negatives → Contrastive training
```

**Step 1: Doc → Query** (`doc_to_query.py`)
- LLM (Llama 70B) generates reasoning-intensive questions from documents
- Prompt (`DOC2HARD_QUERY`): questions must be standalone, require higher-order thinking (analysis/evaluation/synthesis), NOT fact recall, NOT reading comprehension
- Questions use domain-specific scenarios with technical terminology

**Step 2: Hard negative mining** (`hard_negative_mining.py`)
- BM25 index over all documents (Lucene BM25, k1=0.9, b=0.4)
- For each generated query, retrieve top-1000 docs
- Hard negatives taken from **rank 20+** (avoids false negatives in top results)

**Step 3: Reasoning generation** (`generate_reasoning.py`)
- For each query, LLM generates chain-of-thought reasoning
- Prompt: "Identify the problem → Think step by step → Describe relevant info → Draft answer"
- Used as query prefix during training

**Final format**: `{query, pos: [document], neg: [hard_negatives]}` — contrastive triplets

### Our adaptation for PMC-Patients

Same pipeline, applied to clinical patient-to-patient retrieval:

1. **Documents** = 250K patient case summaries from PMC-Patients V2
2. **Query generation** = DeepSeek V3.2 via OpenRouter (~$0.04/M input, cheapest option)
   - Generate reasoning-intensive clinical queries from each patient case
   - "A patient presents with X, Y, Z — which prior cases are most relevant?"
3. **Hard negative mining** = BM25 over all 250K patients, negatives from rank 20+
4. **Training** = PyLate contrastive loss with temperature=0.02

### Why this approach over Knowledge Distillation

| | ReasonIR (contrastive) | KD (distillation) |
|---|---|---|
| LLM calls | 1 per doc (generate query) | 1 per query × N candidates (score each) |
| Cost for 2000 docs | ~$0.10 | ~$0.60+ |
| Training signal | Binary (pos/neg) | Continuous (0-1) |
| Proven for this model? | Yes — this is how it was trained | No |
| Complexity | Simple | More complex |

### Key hyperparameters from original ReasonIR

- `queries_per_doc`: 1
- `hard_neg_start_index`: 20 (skip top-20 BM25 results)
- `temperature`: 0 (deterministic generation)
- Contrastive loss temperature: 0.02
- Learning rate: 3e-6
- Training: contrastive with `losses.Contrastive` (PyLate)

"""
ReasonIR-style Synthetic Data Pipeline for PMC-Patients

Follows the exact methodology from facebook/ReasonIR:
1. Load patient cases as documents
2. LLM generates reasoning-intensive clinical queries from each patient
3. BM25 mines hard negatives (rank 20+)
4. Train ColBERT with contrastive loss
"""

import json
import os
import re
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
N_DOCS = 2000                # number of patient cases to generate queries for
QUERIES_PER_DOC = 1          # queries per patient case
HARD_NEG_START_RANK = 20     # skip top-20 BM25 results (avoid false negatives)
N_HARD_NEGS = 7              # hard negatives per query
OUTPUT_DIR = Path("reasonir_data")
LLM_MODEL = "deepseek/deepseek-v3.2"
LLM_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are tasked with generating {n} reasoning-intensive clinical query from a patient case. The query must be:

1. STANDALONE — meaningful without the original case
2. Present a clinical scenario requiring reasoning to match similar cases
3. Use medical terminology naturally
4. Require analysis of symptoms, diagnosis, treatment patterns — NOT simple keyword matching
5. Be answerable by finding clinically similar patient cases

DO NOT reference the source case directly. Frame as a new clinical question/scenario.

Return JSON:
```json
{{"queries": ["query1"]}}
```"""

USER_PROMPT = """Patient case:

<case>
{patient_text}
</case>

Generate the clinical reasoning query."""


def step1_load_patients():
    """Load 250K patient cases."""
    from huggingface_hub import hf_hub_download

    print("Loading PMC-Patients V2...")
    path = hf_hub_download("zhengyun21/PMC-Patients", "PMC-Patients-V2.json", repo_type="dataset")
    with open(path) as f:
        patients = json.load(f)

    uid_to_text = {}
    for p in patients:
        uid_to_text[p["patient_uid"]] = f"{p['title']}. {p['patient']}"

    print(f"  {len(uid_to_text)} patients loaded")
    return uid_to_text


def step2_build_bm25(uid_to_text):
    """Build BM25 index over all patients."""
    from rank_bm25 import BM25Okapi

    print("Building BM25 index...")
    uids = list(uid_to_text.keys())
    tokenized = [uid_to_text[uid].lower().split() for uid in uids]
    bm25 = BM25Okapi(tokenized)
    print(f"  Indexed {len(uids)} documents")
    return bm25, uids


def step3_sample_documents(uid_to_text):
    """Sample diverse patient cases (one per PMID to avoid clusters)."""
    random.seed(42)
    seen_pmids = set()
    eligible = []
    for uid, text in uid_to_text.items():
        pmid = uid.split("-")[0]
        # Skip very short cases
        if len(text) < 200:
            continue
        if pmid not in seen_pmids:
            seen_pmids.add(pmid)
            eligible.append(uid)

    random.shuffle(eligible)
    selected = eligible[:N_DOCS]
    print(f"  Selected {len(selected)} diverse patient cases")
    return selected


def step4_generate_queries(uid_to_text, selected_uids):
    """LLM generates reasoning-intensive queries from patient cases (parallel)."""
    import asyncio
    from openai import AsyncOpenAI

    CONCURRENCY = 50  # parallel requests

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    queries_path = OUTPUT_DIR / "generated_queries.jsonl"

    # Resume
    done_uids = set()
    if queries_path.exists():
        with open(queries_path) as f:
            for line in f:
                done_uids.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done_uids)} already generated")

    remaining = [uid for uid in selected_uids if uid not in done_uids]
    print(f"Generating queries for {len(remaining)} patients with {LLM_MODEL} (concurrency={CONCURRENCY})...")

    sys_prompt = SYSTEM_PROMPT.format(n=QUERIES_PER_DOC)
    total_cost = 0.0
    total_done = 0
    total_errors = 0
    lock = asyncio.Lock()

    async def process_one(uid, sem, f):
        nonlocal total_cost, total_done, total_errors
        async with sem:
            text = uid_to_text[uid]
            user_msg = USER_PROMPT.format(patient_text=text[:1500])

            try:
                resp = await client.chat.completions.create(
                    model=LLM_MODEL,
                    max_tokens=300,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )

                content = resp.choices[0].message.content or ""
                match = re.search(r"\{[\s\S]*\}", content)
                if not match:
                    total_errors += 1
                    return

                parsed = json.loads(match.group())
                queries = parsed.get("queries", [])
                if not queries:
                    total_errors += 1
                    return

                result = {
                    "uid": uid,
                    "document": text[:2000],
                    "queries": queries[:QUERIES_PER_DOC],
                }
                async with lock:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    cost = getattr(resp.usage, "cost", 0) or 0
                    total_cost += cost
                    total_done += 1
                    if total_done % 100 == 0:
                        print(f"  {total_done}/{len(remaining)} done, {total_errors} errors (${total_cost:.4f})")

            except Exception as e:
                total_errors += 1
                if total_errors % 20 == 0:
                    print(f"  {total_errors} errors so far, latest: {e}")

    async def run_all():
        sem = asyncio.Semaphore(CONCURRENCY)
        # Use sync file handle (writes are small and serialized by lock)
        f = open(queries_path, "a")
        tasks = [process_one(uid, sem, f) for uid in remaining]
        await asyncio.gather(*tasks)
        f.close()

    asyncio.run(run_all())
    print(f"Query generation complete. {total_done} succeeded, {total_errors} errors. Cost: ${total_cost:.4f}")


def step5_mine_hard_negatives(uid_to_text, bm25, uids):
    """BM25 hard negative mining (rank 20+ to avoid false negatives)."""
    queries_path = OUTPUT_DIR / "generated_queries.jsonl"
    training_path = OUTPUT_DIR / "training_data.jsonl"

    # Load generated queries
    generated = []
    with open(queries_path) as f:
        for line in f:
            generated.append(json.loads(line))

    print(f"Mining hard negatives for {len(generated)} examples...")
    uid_set = set(uids)
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}

    training_data = []
    for i, entry in enumerate(generated):
        doc_uid = entry["uid"]
        doc_text = entry["document"]
        doc_pmid = doc_uid.split("-")[0]

        for query in entry["queries"]:
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)
            top_indices = scores.argsort()[::-1]

            # Hard negatives: rank 20+, skip same PMID
            hard_negs = []
            rank = 0
            for idx in top_indices:
                neg_uid = uids[idx]
                neg_pmid = neg_uid.split("-")[0]
                if neg_uid == doc_uid or neg_pmid == doc_pmid:
                    continue
                rank += 1
                if rank >= HARD_NEG_START_RANK:
                    hard_negs.append(uid_to_text[neg_uid])
                    if len(hard_negs) >= N_HARD_NEGS:
                        break

            training_data.append({
                "query": query,
                "pos": [doc_text],
                "neg": hard_negs,
            })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(generated)} done")

    # Save
    with open(training_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(training_data)} training examples to {training_path}")
    return training_data


def step6_format_for_pylate(training_data=None):
    """Convert to HuggingFace Dataset for PyLate contrastive training."""
    from datasets import Dataset

    if training_data is None:
        training_path = OUTPUT_DIR / "training_data.jsonl"
        training_data = []
        with open(training_path) as f:
            for line in f:
                training_data.append(json.loads(line))

    # PyLate contrastive format: query, positive, negative
    rows = []
    for item in training_data:
        for pos in item["pos"]:
            for neg in item["neg"]:
                rows.append({
                    "query": item["query"],
                    "positive": pos,
                    "negative": neg,
                })

    ds = Dataset.from_list(rows)
    ds_path = str(OUTPUT_DIR / "dataset")
    ds.save_to_disk(ds_path)
    print(f"Saved {len(rows)} triplets to {ds_path}")
    return ds


def step7_train():
    """Fine-tune ColBERT with contrastive loss."""
    from datasets import load_from_disk
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from pylate import losses, models, utils, evaluation

    model = models.ColBERT(model_name_or_path="lightonai/Reason-ModernColBERT")

    dataset = load_from_disk(str(OUTPUT_DIR / "dataset"))
    splits = dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    train_loss = losses.Contrastive(model=model, temperature=0.02)

    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    args = SentenceTransformerTrainingArguments(
        output_dir="output/reason-colbert-clinical",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        fp16=True,
        learning_rate=3e-6,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        run_name="reason-colbert-clinical",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    trainer.train()
    model.save_pretrained("output/reason-colbert-clinical/final")
    print("Training complete!")


if __name__ == "__main__":
    import sys

    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step == "all":
        uid_to_text = step1_load_patients()
        bm25, uids = step2_build_bm25(uid_to_text)
        selected = step3_sample_documents(uid_to_text)
        step4_generate_queries(uid_to_text, selected)
        training_data = step5_mine_hard_negatives(uid_to_text, bm25, uids)
        step6_format_for_pylate(training_data)
    elif step == "generate":
        uid_to_text = step1_load_patients()
        selected = step3_sample_documents(uid_to_text)
        step4_generate_queries(uid_to_text, selected)
    elif step == "mine":
        uid_to_text = step1_load_patients()
        bm25, uids = step2_build_bm25(uid_to_text)
        training_data = step5_mine_hard_negatives(uid_to_text, bm25, uids)
        step6_format_for_pylate(training_data)
    elif step == "train":
        step7_train()
    else:
        print(f"Unknown step: {step}. Use: all, generate, mine, train")

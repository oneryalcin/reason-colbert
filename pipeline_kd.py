"""
Knowledge Distillation Pipeline for PMC-Patients Clinical Retrieval

Task: Patient-to-Patient similarity retrieval
  Given a patient case, find clinically similar patient cases.

Pipeline:
1. Load PMC-Patients corpus (250K patients)
2. Build BM25 index over patient texts
3. For each query patient, BM25 retrieve top-K candidates
4. Use Claude to score clinical similarity (continuous 0-1)
5. Save as PyLate KD training format
6. Fine-tune ColBERT with distillation loss
"""

import json
import os
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
N_TRAIN_QUERIES = 2000       # queries to generate KD data for
N_CANDIDATES = 20            # BM25 candidates per query
OUTPUT_DIR = Path("kd_data")
LLM_MODEL = "minimax/minimax-m2.7"
LLM_BASE_URL = "https://openrouter.ai/api/v1"


def step1_load_patients():
    """Load 250K patient cases from PMC-Patients V2."""
    from huggingface_hub import hf_hub_download

    print("Loading PMC-Patients V2...")
    path = hf_hub_download("zhengyun21/PMC-Patients", "PMC-Patients-V2.json", repo_type="dataset")
    with open(path) as f:
        patients_list = json.load(f)

    uid_to_text = {}
    for p in patients_list:
        uid_to_text[p["patient_uid"]] = f"{p['title']}. {p['patient']}"

    print(f"  {len(uid_to_text)} patients loaded")
    return uid_to_text


def step2_build_bm25(uid_to_text):
    """Build BM25 index over all patient texts."""
    from rank_bm25 import BM25Okapi

    print("Building BM25 index over 250K patients...")
    uids = list(uid_to_text.keys())
    tokenized = [uid_to_text[uid].lower().split() for uid in uids]
    bm25 = BM25Okapi(tokenized)
    print(f"  Indexed {len(uids)} documents")
    return bm25, uids


def step3_generate_candidates(uid_to_text, bm25, uids):
    """For each query patient, BM25-retrieve top-K similar patients."""
    random.seed(42)

    # Sample diverse queries — pick from different PMIDs to avoid same-paper clusters
    seen_pmids = set()
    eligible = []
    for uid in uids:
        pmid = uid.split("-")[0]
        if pmid not in seen_pmids:
            seen_pmids.add(pmid)
            eligible.append(uid)

    random.shuffle(eligible)
    query_uids = eligible[:N_TRAIN_QUERIES]

    print(f"Generating {N_CANDIDATES} BM25 candidates for {len(query_uids)} queries...")
    candidates = []
    for i, quid in enumerate(query_uids):
        query_text = uid_to_text[quid]
        query_tokens = query_text.lower().split()
        scores = bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1]

        # Skip self and same-paper patients
        query_pmid = quid.split("-")[0]
        cands = []
        for idx in top_indices:
            cuid = uids[idx]
            if cuid == quid or cuid.split("-")[0] == query_pmid:
                continue
            cands.append({"uid": cuid, "text": uid_to_text[cuid]})
            if len(cands) >= N_CANDIDATES:
                break

        candidates.append({
            "query_uid": quid,
            "query_text": query_text,
            "candidates": cands,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(query_uids)} queries done")

    print(f"  Generated {len(candidates)} candidate sets")
    return candidates


def step4_llm_score(candidates):
    """Use LLM via OpenRouter to score clinical similarity."""
    import re
    from openai import OpenAI

    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    scored_path = OUTPUT_DIR / "scored_candidates.jsonl"

    # Resume from where we left off
    done_queries = set()
    if scored_path.exists():
        with open(scored_path) as f:
            for line in f:
                done_queries.add(json.loads(line)["query_uid"])
        print(f"Resuming: {len(done_queries)} queries already scored")

    system_prompt = """You are a clinical similarity judge. Given a query patient case and candidate patient cases, rate their clinical similarity on a scale from 0.0 to 1.0.

Consider (in order of importance):
1. Same or closely related diagnosis/condition
2. Similar symptoms and clinical presentation
3. Similar treatment approach or clinical course
4. Similar lab values, imaging, or procedures
5. Similar demographics only if clinically relevant

Score guide:
- 0.9-1.0: Same condition, very similar presentation
- 0.7-0.8: Same disease area, related presentation
- 0.4-0.6: Some clinical overlap but different primary condition
- 0.1-0.3: Minimal clinical relevance
- 0.0: Completely unrelated

End your response with a JSON array, one object per candidate in order:
[{"score": <float>, "reason": "<one sentence>"}]"""

    remaining = [c for c in candidates if c["query_uid"] not in done_queries]
    print(f"Scoring {len(remaining)} queries with {LLM_MODEL}...")
    total_cost = 0.0

    with open(scored_path, "a") as f:
        for i, entry in enumerate(remaining):
            candidate_block = "\n\n".join(
                f"[Candidate {j+1}]: {cand['text'][:800]}"
                for j, cand in enumerate(entry["candidates"])
            )

            user_msg = f"[Query Patient]: {entry['query_text'][:800]}\n\n{candidate_block}"

            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    max_tokens=2000,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )

                # Handle reasoning models (content in model_extra.reasoning)
                msg = resp.choices[0].message
                text = msg.content
                if not text and hasattr(msg, "model_extra"):
                    text = msg.model_extra.get("reasoning", "")

                if not text:
                    print(f"  Empty response for {entry['query_uid']}")
                    continue

                # Extract last valid JSON array from response
                scores = None
                for match in reversed(re.findall(r"\[[\s\S]*?\]", text)):
                    try:
                        parsed = json.loads(match)
                        if parsed and isinstance(parsed[0], dict) and "score" in parsed[0]:
                            scores = parsed
                            break
                    except (json.JSONDecodeError, IndexError):
                        continue

                if not scores:
                    print(f"  No JSON found for {entry['query_uid']}")
                    continue

                scored = []
                for j, cand in enumerate(entry["candidates"]):
                    if j < len(scores):
                        scored.append({
                            "uid": cand["uid"],
                            "score": float(scores[j]["score"]),
                            "reason": scores[j].get("reason", ""),
                        })

                result = {
                    "query_uid": entry["query_uid"],
                    "query_text": entry["query_text"][:2000],
                    "candidates": scored,
                }
                f.write(json.dumps(result) + "\n")
                f.flush()

                cost = getattr(resp.usage, "cost", 0) or 0
                total_cost += cost

            except Exception as e:
                print(f"  Error on query {entry['query_uid']}: {e}")
                continue

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(remaining)} scored (${total_cost:.4f} so far)")

    print(f"Scoring complete. Total cost: ${total_cost:.4f}. Saved to {scored_path}")


def step5_format_for_pylate():
    """Convert scored data to PyLate KD training format."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    scored_path = OUTPUT_DIR / "scored_candidates.jsonl"

    # Reload full patient texts
    path = hf_hub_download("zhengyun21/PMC-Patients", "PMC-Patients-V2.json", repo_type="dataset")
    with open(path) as f:
        patients_list = json.load(f)
    full_uid_to_text = {p["patient_uid"]: f"{p['title']}. {p['patient']}" for p in patients_list}

    training_data = []
    all_query_ids = set()
    all_doc_ids = set()

    with open(scored_path) as f:
        for line in f:
            entry = json.loads(line)
            quid = entry["query_uid"]
            doc_ids = [c["uid"] for c in entry["candidates"]]
            scores = [c["score"] for c in entry["candidates"]]

            if doc_ids:
                training_data.append({
                    "query_id": quid,
                    "document_ids": doc_ids,
                    "scores": scores,
                })
                all_query_ids.add(quid)
                all_doc_ids.update(doc_ids)

    train_ds = Dataset.from_list(training_data)
    queries = Dataset.from_list([
        {"query_id": uid, "text": full_uid_to_text.get(uid, "")}
        for uid in all_query_ids
    ])
    documents = Dataset.from_list([
        {"document_id": uid, "text": full_uid_to_text.get(uid, "")}
        for uid in all_doc_ids | all_query_ids
    ])

    train_ds.save_to_disk(str(OUTPUT_DIR / "train"))
    queries.save_to_disk(str(OUTPUT_DIR / "queries"))
    documents.save_to_disk(str(OUTPUT_DIR / "documents"))

    print(f"Saved PyLate KD data:")
    print(f"  {len(training_data)} training examples")
    print(f"  {len(queries)} queries, {len(documents)} documents")


def step6_train():
    """Fine-tune ColBERT with knowledge distillation."""
    from datasets import load_from_disk
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from pylate import losses, models, utils

    model = models.ColBERT(model_name_or_path="lightonai/Reason-ModernColBERT")

    train = load_from_disk(str(OUTPUT_DIR / "train"))
    queries = load_from_disk(str(OUTPUT_DIR / "queries"))
    documents = load_from_disk(str(OUTPUT_DIR / "documents"))

    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )

    train_loss = losses.Distillation(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir="output/reason-colbert-clinical",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        fp16=True,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        run_name="reason-colbert-clinical-kd",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
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
        candidates = step3_generate_candidates(uid_to_text, bm25, uids)

        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(OUTPUT_DIR / "candidates.json", "w") as f:
            json.dump(candidates, f)
        print(f"Saved {len(candidates)} candidate sets")

        step4_llm_score(candidates)
        step5_format_for_pylate()
    elif step == "score":
        with open(OUTPUT_DIR / "candidates.json") as f:
            candidates = json.load(f)
        step4_llm_score(candidates)
    elif step == "format":
        step5_format_for_pylate()
    elif step == "train":
        step6_train()
    else:
        print(f"Unknown step: {step}. Use: all, score, format, train")

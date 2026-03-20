import modal
import time
import os

app = modal.App("reason-colbert-l40s")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch>=2.2.0", gpu="l40s")
    .uv_pip_install("pylate>=1.4.0", "datasets")
)

MODEL_ID = "lightonai/Reason-ModernColBERT"
MODEL_CACHE = "/model-cache/reason-colbert"

vol = modal.Volume.from_name("reason-colbert-model", create_if_missing=True)


@app.cls(
    image=image,
    gpu="l40s",
    volumes={"/model-cache": vol},
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=16)
class ColBERTService:
    @modal.enter()
    def load_model(self):
        import torch
        from pylate import models

        if os.path.exists(os.path.join(MODEL_CACHE, "config.json")):
            load_path = MODEL_CACHE
        else:
            load_path = MODEL_ID

        self.model = models.ColBERT(
            model_name_or_path=load_path,
            device="cuda",
            model_kwargs={"attn_implementation": "sdpa"},
        )
        self.model.half()

        if load_path == MODEL_ID:
            self.model.save_pretrained(MODEL_CACHE)
            vol.commit()

        for _ in range(3):
            self.model.encode(["warmup"], is_query=True)
            self.model.encode(["warmup"], is_query=False)
        torch.cuda.synchronize()

    @modal.method()
    def throughput_test(self, n_docs: int = 1000):
        import torch
        import statistics
        from datasets import load_dataset

        ds = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
        texts = []
        while len(texts) < n_docs:
            for row in ds:
                texts.append(row["title"] + " " + row["text"])
                if len(texts) >= n_docs:
                    break

        char_lens = [len(t) for t in texts]
        tok_sample = self.model.tokenize(texts[:100])
        avg_tokens = tok_sample["input_ids"].shape[1]

        results = {
            "gpu": torch.cuda.get_device_name(0),
            "n_docs": n_docs,
            "avg_chars": round(statistics.mean(char_lens)),
            "est_avg_tokens": avg_tokens,
        }

        for bs in [16, 32, 64, 128, 256]:
            self.model.encode(texts[:bs], is_query=False, batch_size=bs)
            torch.cuda.synchronize()
            t0 = time.time()
            self.model.encode(texts[:n_docs], is_query=False, batch_size=bs)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            docs_per_sec = n_docs / elapsed
            results[f"bs{bs}"] = {
                "total_sec": round(elapsed, 2),
                "docs_per_sec": round(docs_per_sec, 1),
                "ms_per_doc": round(elapsed / n_docs * 1000, 2),
            }

        return results

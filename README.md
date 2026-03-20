# Reason-ModernColBERT Benchmarks

Evaluation and deployment benchmarks for [lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT) — a 150M parameter ColBERT late-interaction retrieval model trained on ReasonIR data.

## Model

- **Architecture**: ModernBERT (22 layers, 768 hidden, 12 heads) + 128-dim Dense projection
- **Parameters**: 149M
- **Max sequence**: 8192 tokens
- **Attention**: Sliding window (128) with global attention every 3rd layer
- **Library**: [PyLate](https://github.com/lightonai/pylate)

## Setup

```bash
uv sync
# Download model
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('lightonai/Reason-ModernColBERT', local_dir='./model')
"
```

## Local test

```bash
uv run python test_model.py
```

## Modal deployment

```bash
uv run modal deploy modal_app.py
uv run python test_modal.py
```

## Benchmarks

All benchmarks use real documents from [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) (medical/scientific text, ~128 tokens avg, ~1627 chars avg).

### Local: MPS vs ONNX (Apple Silicon M-series)

| 10 docs | MPS | ONNX fp32 | ONNX int8 |
|---|---|---|---|
| Short ~7tok | 6.9ms/doc | **2.3ms/doc** | **2.2ms/doc** |
| Medium ~140tok | **13.6ms/doc** | 44.4ms/doc | 33.5ms/doc |
| Long ~700tok | **217.8ms/doc** | 398.1ms/doc | 415.2ms/doc |

MPS wins on medium/long docs. ONNX wins on short docs (less overhead).

### GPU: dtype comparison (L4, 500 docs, batch_size=16)

| Config | docs/sec | ms/doc | vs FP32 |
|---|---|---|---|
| FP32 | 51 | 20.1 | 1x |
| **FP16** | **168** | **5.9** | **3.3x** |
| BF16 | 167 | 6.0 | 3.3x |
| torch.compile + FP16 | 2.9 | 342 | N/A (compilation overhead) |

FP16 via `.half()` is the clear winner. `torch_dtype` kwarg in `model_kwargs` is ignored by PyLate — must call `.half()` explicitly.

### GPU: L4 vs L40S throughput (1000 docs, FP16 + SDPA)

| Batch Size | L4 (docs/sec) | L40S (docs/sec) | Speedup |
|---|---|---|---|
| 16 | 192 | **486** | 2.5x |
| 32 | 161 | **499** | 3.1x |
| 64 | 128 | 402 | 3.1x |
| 128 | 97 | 298 | 3.1x |
| 256 | 65 | 208 | 3.2x |

L40S peaks at **~500 docs/sec** (batch_size=32) = **1.8M docs/hour**.

Cost per doc is roughly equivalent (L40S ~3x price, ~3x throughput). L40S wins on latency (2ms vs 5ms/doc).

### GPU: L4 vs local MPS

| 10 docs | MPS (Apple Silicon) | L4 GPU | Speedup |
|---|---|---|---|
| Short ~7tok | 6.9ms/doc | 2.64ms/doc | 2.6x |
| Medium ~140tok | 13.6ms/doc | 2.64ms/doc | 5.2x |
| Long ~700tok | 217.8ms/doc | 13.01ms/doc | 16.7x |

## Key findings

1. **FP16 is essential** — 3.3x speedup over FP32 on GPU with no quality loss for inference
2. **SDPA ≈ Flash Attention 2** — PyTorch's native SDPA uses flash kernels on Ampere+ GPUs; no need to compile flash-attn
3. **torch.compile is a trap** for small models — compilation overhead (minutes) dwarfs any runtime gain
4. **Batch size 16-32 is optimal** — smaller batches = less padding waste for variable-length docs
5. **FP16 > BF16 for inference** — same speed, but FP16 has 3 extra precision bits for scoring
6. **Quantization (INT8/FP8) not worth it** — model is only 0.3GB in FP16, no memory pressure
7. **L40S is 3x faster than L4** at same cost/doc — pick L40S for latency-sensitive workloads

## Files

| File | Description |
|---|---|
| `test_model.py` | Local semantic reranking test (MPS) |
| `modal_app.py` | Modal deployment (L4, FP16, SDPA, Volume caching) |
| `modal_app_l40s.py` | Modal deployment (L40S variant for benchmarking) |
| `test_modal.py` | Remote benchmark + semantic test client |

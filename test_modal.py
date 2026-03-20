import modal
import time

ColBERTService = modal.Cls.from_name("reason-colbert", "ColBERTService")


def main():
    svc = ColBERTService()

    # Benchmark
    print("=== GPU Benchmark (L4 + SDPA + FP16) ===")
    results = svc.benchmark.remote()
    for name, r in results.items():
        print(f"  {name:15s} x10: {r['total_ms']:6.1f}ms  ({r['per_doc_ms']:.2f}ms/doc)")

    # Semantic test
    print("\n=== Semantic Rerank Test ===")
    queries = [
        "Why do leaves appear green?",
        "What happens when you drop something?",
        "How does the body fight off illness?",
        "Why is the ocean salty?",
    ]
    docs = [
        "Chlorophyll pigments absorb red and blue wavelengths, reflecting the remaining spectrum.",
        "Objects accelerate toward Earth at 9.8 m/s² due to gravitational pull.",
        "White blood cells identify and destroy pathogens through antibody response.",
        "Rivers carry dissolved sodium and chloride ions into seas, accumulating over millennia.",
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
        "Compound interest calculates returns on both principal and previously earned gains.",
    ]

    t0 = time.time()
    results = svc.rerank.remote(queries, docs, list(range(len(docs))))
    latency = (time.time() - t0) * 1000

    for i, q in enumerate(queries):
        print(f"\nQuery: {q}")
        for r in results[i]:
            print(f"  {r['score']:.4f}  {docs[r['id']]}")

    print(f"\nEnd-to-end (incl. network): {latency:.0f}ms")


if __name__ == "__main__":
    main()

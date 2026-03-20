from pylate import models, rank

model = models.ColBERT(model_name_or_path="./model", device="mps")

# Queries have zero word overlap with their target docs
queries = [
    "Why do leaves appear green?",                  # expect: chlorophyll absorbs red/blue
    "What happens when you drop something?",        # expect: gravitational acceleration
    "How does the body fight off illness?",          # expect: immune system / white blood cells
    "Why is the ocean salty?",                       # expect: dissolved minerals from rivers
]

docs = [
    "Chlorophyll pigments absorb red and blue wavelengths, reflecting the remaining spectrum.",
    "Objects accelerate toward Earth at 9.8 m/s² due to gravitational pull.",
    "White blood cells identify and destroy pathogens through antibody response.",
    "Rivers carry dissolved sodium and chloride ions into seas, accumulating over millennia.",
    "The Eiffel Tower was completed in 1889 for the World's Fair.",
    "Compound interest calculates returns on both principal and previously earned gains.",
]

doc_ids = list(range(len(docs)))
q_emb = model.encode(queries, is_query=True)
d_emb = model.encode(docs, is_query=False)

results = rank.rerank(
    documents_ids=[doc_ids] * len(queries),
    queries_embeddings=q_emb,
    documents_embeddings=[d_emb] * len(queries),
)

for i, q in enumerate(queries):
    print(f"\nQuery: {q}")
    for r in results[i]:
        print(f"  {r['score']:.4f}  {docs[r['id']]}")

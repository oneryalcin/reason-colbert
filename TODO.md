# TODO

## Performance

- [ ] Replace `rank_bm25` with `tantivy-py` (Rust BM25) for hard negative mining. Current BM25 over 250K docs takes minutes per query batch. Tantivy would be near-instant. `pip install tantivy`. ReasonIR itself uses Lucene BM25 via pyserini — tantivy is the Rust equivalent.

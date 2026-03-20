# TODO

## Performance

- [x] Replace `rank_bm25` with `tantivy-py` (Rust BM25) for hard negative mining. Index 250K docs in 4s, query in 1ms vs 500ms. 1995 queries: ~2s vs ~17min.

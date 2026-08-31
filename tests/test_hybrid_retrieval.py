from services.agent_runtime.app.hybrid_retrieval import (
    EVAL_CORPUS,
    EVAL_QUERIES,
    HARD_QUERIES,
    HashedNgramEmbedding,
    evaluate_hard_retrieval,
    evaluate_modes,
    evaluate_zero_overlap,
    lexical_overlap,
    retrieve,
)


def test_hybrid_rerank_beats_keyword_on_local_eval_set() -> None:
    report = evaluate_modes(EVAL_QUERIES, EVAL_CORPUS, k=3)
    assert report["keyword"]["queries"] == 19
    assert report["hybrid_rerank"]["recall_at_k"] >= 0.85
    assert report["hybrid_rerank"]["mrr"] >= report["keyword"]["mrr"]
    assert report["hybrid_late_interaction"]["recall_at_k"] >= 0.85
    assert report["hybrid_late_interaction"]["mrr"] >= report["keyword"]["mrr"]
    assert "hybrid_cross_encoder" in report
    assert report["hybrid_cross_encoder"]["queries"] == 19


def test_zero_overlap_queries_have_no_shared_tokens() -> None:
    for item in EVAL_QUERIES:
        if item.get("family") != "zero_overlap":
            continue
        passage = next(row for row in EVAL_CORPUS if row.passage_id == item["relevant"][0])
        assert lexical_overlap(item["query"], f"{passage.title} {passage.text}") == 0


def test_minilm_vector_path_recovers_zero_overlap_semantics() -> None:
    gap = evaluate_zero_overlap(k=3)
    assert gap["vector"]["recall_at_k"] > gap["keyword"]["recall_at_k"]
    assert gap["hybrid"]["recall_at_k"] >= gap["vector"]["recall_at_k"]
    hits = retrieve("quote grounded summary rule", EVAL_CORPUS, limit=1, mode="vector")
    assert hits[0].passage.passage_id == "p-citation"


def test_lexical_query_keeps_exact_match_first() -> None:
    hits = retrieve("cited passage_id", EVAL_CORPUS, limit=3, mode="bm25")
    assert hits[0].passage.passage_id == "p-citation"


def test_hashed_embedding_is_lexical_ablation_only() -> None:
    backend = HashedNgramEmbedding()
    left = backend.embed("Reciprocal Rank Fusion")
    right = backend.embed("Reciprocal Rank Fusion")
    assert left == right
    hashed = evaluate_zero_overlap(k=3, embedding_backend=backend)
    semantic = evaluate_zero_overlap(k=3)
    assert semantic["vector"]["recall_at_k"] == 1.0
    assert hashed["vector"]["recall_at_k"] < semantic["vector"]["recall_at_k"]
    assert (
        retrieve("quote grounded summary rule", EVAL_CORPUS, limit=1, mode="vector")[
            0
        ].passage.passage_id
        == "p-citation"
    )


def test_cross_encoder_ranks_lexical_match_first() -> None:
    hits = retrieve("cited passage_id", EVAL_CORPUS, limit=1, mode="hybrid_cross_encoder")
    assert hits[0].passage.passage_id == "p-citation"
    assert retrieve.__kwdefaults__ is not None
    assert retrieve.__kwdefaults__["mode"] == "hybrid_late_interaction"


def test_hard_retrieval_queries_share_no_tokens_with_targets() -> None:
    for item in HARD_QUERIES:
        passage = next(row for row in EVAL_CORPUS if row.passage_id == item["relevant"][0])
        assert lexical_overlap(item["query"], f"{passage.title} {passage.text}") == 0
    hard = evaluate_hard_retrieval(k=3)
    assert hard["keyword"]["recall_at_k"] < hard["vector"]["recall_at_k"]
    assert hard["hybrid_cross_encoder"]["queries"] == float(len(HARD_QUERIES))

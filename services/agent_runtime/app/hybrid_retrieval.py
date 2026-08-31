from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_RRF_K = 60
_EMBED_DIM = 256
_CHUNK_CHARS = 400
_CHUNK_OVERLAP = 60
RetrievalMode = Literal[
    "keyword",
    "bm25",
    "vector",
    "hybrid",
    "hybrid_rerank",
    "hybrid_late_interaction",
    "hybrid_cross_encoder",
]
_NEURAL_CONFIDENCE_GAP = 0.18
_WINDOW_CHARS = 96


class EmbeddingBackend(Protocol):
    """Swap-in contract for MiniLM or a test double."""

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class Passage:
    passage_id: str
    source_id: str
    kind: str
    title: str
    text: str


@dataclass
class RankedHit:
    passage: Passage
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    def as_tool_result(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage.passage_id,
            "source_id": self.passage.source_id,
            "kind": self.passage.kind,
            "title": self.passage.title,
            "snippet": self.passage.text[:500],
            "score": round(self.score, 6),
            "ranks": self.ranks,
        }


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def char_ngrams(text: str, size: int = 3) -> list[str]:
    folded = re.sub(r"\s+", " ", text.lower()).strip()
    if len(folded) < size:
        return [folded] if folded else []
    return [folded[index : index + size] for index in range(len(folded) - size + 1)]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return vector
    return [value / norm for value in vector]


class HashedNgramEmbedding:
    """Lexical ablation only. This is not the production vector path."""

    def __init__(self, dimension: int = _EMBED_DIM) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        features = char_ngrams(text, 3) + tokenize(text)
        if not features:
            return vector
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


_MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM_GCS = (
    "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz"
)
_minilm_model: Any = None


def _minilm_cache_dir() -> str:
    override = os.environ.get("FASTEMBED_CACHE_PATH")
    path = Path(override) if override else Path.home() / ".cache" / "fastembed"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _minilm_model_instance() -> Any:
    """Load MiniLM from the Qdrant ONNX tarball. Avoids a Hugging Face Hub hang."""
    global _minilm_model
    if _minilm_model is None:
        from fastembed import TextEmbedding
        from fastembed.common.model_management import ModelManagement

        model_dir = ModelManagement.retrieve_model_gcs(
            _MINILM_NAME,
            _MINILM_GCS,
            _minilm_cache_dir(),
            deprecated_tar_struct=True,
        )
        _minilm_model = TextEmbedding(model_name=_MINILM_NAME, specific_model_path=str(model_dir))
    return _minilm_model


class MiniLMEmbedding:
    """all-MiniLM-L6-v2 on CPU via ONNX. This is the semantic vector path."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, ...]] = {}

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            vectors = list(_minilm_model_instance().embed(missing))
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = tuple(_l2_normalize([float(value) for value in vector]))
        return [list(self._cache[text]) for text in texts]


_DEFAULT_MINILM: MiniLMEmbedding | None = None


def default_embedding() -> MiniLMEmbedding:
    global _DEFAULT_MINILM
    if _DEFAULT_MINILM is None:
        _DEFAULT_MINILM = MiniLMEmbedding()
    return _DEFAULT_MINILM


def cosine(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


class BM25Index:
    """Okapi BM25 with the always-non-negative log((N-df+0.5)/(df+0.5)+1) IDF."""

    def __init__(
        self,
        documents: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_len = [len(tokens) or 1 for tokens in documents]
        self.avgdl = sum(self.doc_len) / max(1, len(documents))
        df: Counter[str] = Counter()
        self.tfs: list[Counter[str]] = []
        for tokens in documents:
            counts = Counter(tokens)
            self.tfs.append(counts)
            df.update(counts.keys())
        n_docs = max(1, len(documents))
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.documents)
        for index, tf in enumerate(self.tfs):
            length_norm = 1.0 - self.b + self.b * self.doc_len[index] / self.avgdl
            total = 0.0
            for term in query_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                total += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * length_norm)
            scores[index] = total
        return scores


def keyword_scores(query: str, passages: list[Passage]) -> list[float]:
    needle = query.lower().strip()
    query_tokens = set(tokenize(query))
    scores: list[float] = []
    for passage in passages:
        haystack = f"{passage.title} {passage.text}".lower()
        if needle and needle in haystack:
            scores.append(2.0 + haystack.count(needle))
            continue
        doc_tokens = set(tokenize(haystack))
        overlap = len(query_tokens & doc_tokens)
        scores.append(overlap / max(1, len(query_tokens)))
    return scores


def rrf_fuse(*rankings: list[str], k: int = _RRF_K) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, passage_id in enumerate(ranking, start=1):
            fused[passage_id] += 1.0 / (k + rank)
    return dict(fused)


def _ranked_ids(scores: list[float], passages: list[Passage]) -> list[str]:
    order = sorted(
        range(len(passages)), key=lambda index: (-scores[index], passages[index].passage_id)
    )
    return [passages[index].passage_id for index in order if scores[index] > 0]


class FeatureBasedReranker:
    """Weighted lexical/semantic features. Not a neural cross-encoder."""

    weights = (0.34, 0.22, 0.22, 0.12, 0.10)

    def score_pair(self, query: str, document: str, bm25: float, embedding: float) -> float:
        query_tokens = set(tokenize(query))
        doc_tokens = set(tokenize(document))
        coverage = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
        union = query_tokens | doc_tokens
        jaccard = len(query_tokens & doc_tokens) / max(1, len(union))
        phrase = 1.0 if query.lower().strip() in document.lower() else 0.0
        features = (bm25, embedding, coverage, jaccard, phrase)
        return float(
            sum(weight * value for weight, value in zip(self.weights, features, strict=True))
        )


def passage_windows(text: str) -> list[str]:
    """Sentence-or-char windows for ColBERT-style MaxSim over MiniLM."""
    content = re.sub(r"\s+", " ", text).strip()
    if not content:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", content) if part.strip()]
    if len(sentences) <= 1 and len(content) > _WINDOW_CHARS:
        step = _WINDOW_CHARS - 24
        return [
            content[index : index + _WINDOW_CHARS] for index in range(0, len(content), max(1, step))
        ]
    return sentences or [content]


class MiniLMLateInteractionReranker:
    """Sentence-level MaxSim over MiniLM embeddings.

    Same late-interaction family as ColBERT, but this is not a trained MS MARCO
    cross-encoder and not bge-reranker. It reuses the MiniLM already on disk.
    """

    def score_documents(
        self,
        query: str,
        documents: list[str],
        embedder: EmbeddingBackend,
        query_vector: list[float] | None = None,
    ) -> list[float]:
        q_vec = query_vector if query_vector is not None else embedder.embed(query)
        windows_per_doc = [passage_windows(document) or [document] for document in documents]
        unique_windows = list(
            dict.fromkeys(window for windows in windows_per_doc for window in windows)
        )
        encoded = {
            window: vector
            for window, vector in zip(
                unique_windows, embedder.embed_many(unique_windows), strict=True
            )
        }
        scores: list[float] = []
        for windows in windows_per_doc:
            sims = [cosine(q_vec, encoded[window]) for window in windows]
            top = sorted(sims, reverse=True)
            maxsim = top[0]
            mean_top = sum(top[: min(3, len(top))]) / min(3, len(top))
            scores.append(0.72 * maxsim + 0.28 * mean_top)
        return scores


@dataclass(frozen=True)
class RetrievalTrace:
    reranker: str
    rerank_skipped: bool
    vector_gap: float
    reason: str


@dataclass
class RetrievalOutcome:
    hits: list[RankedHit]
    trace: RetrievalTrace


def lexical_overlap(query: str, document: str) -> int:
    return len(set(tokenize(query)) & set(tokenize(document)))


def chunk_text(source_id: str, kind: str, title: str, text: str) -> list[Passage]:
    content = text.strip()
    if not content:
        return []
    if len(content) <= _CHUNK_CHARS:
        return [Passage(f"{kind}:{source_id}:0", source_id, kind, title, content)]
    passages: list[Passage] = []
    start = 0
    index = 0
    while start < len(content):
        end = min(len(content), start + _CHUNK_CHARS)
        passages.append(
            Passage(f"{kind}:{source_id}:{index}", source_id, kind, title, content[start:end])
        )
        if end >= len(content):
            break
        start = max(end - _CHUNK_OVERLAP, start + 1)
        index += 1
    return passages


async def load_published_passages() -> list[Passage]:
    from sqlalchemy import select

    from services.agent_runtime.app.db import SessionFactory
    from services.core_api.app.models import Paper, PaperSection

    passages: list[Passage] = []
    async with SessionFactory() as session:
        papers = list(await session.scalars(select(Paper).where(Paper.status == "published")))
        sections = list(
            await session.scalars(
                select(PaperSection)
                .join(Paper, Paper.id == PaperSection.paper_id)
                .where(
                    PaperSection.is_published.is_(True),
                    PaperSection.visibility == "public",
                    Paper.status == "published",
                    (PaperSection.publish_at.is_(None) | (PaperSection.publish_at <= datetime.now(UTC))),
                )
            )
        )
    for paper in papers:
        passages.extend(
            chunk_text(str(paper.id), "paper", paper.title, f"{paper.title}. {paper.description}")
        )
    for section in sections:
        passages.extend(chunk_text(str(section.id), "section", section.title, section.content))
    return passages


def vector_confidence_gap(vector_scores: list[float]) -> float:
    ordered = sorted(vector_scores, reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return 1.0
    return float(ordered[0] - ordered[1])


def retrieve(
    query: str,
    passages: list[Passage],
    *,
    limit: int = 5,
    mode: RetrievalMode = "hybrid_late_interaction",
    embedding_backend: EmbeddingBackend | None = None,
    reranker: FeatureBasedReranker | None = None,
    skip_neural_when_confident: bool = False,
) -> list[RankedHit]:
    return retrieve_detailed(
        query,
        passages,
        limit=limit,
        mode=mode,
        embedding_backend=embedding_backend,
        reranker=reranker,
        skip_neural_when_confident=skip_neural_when_confident,
    ).hits


def retrieve_detailed(
    query: str,
    passages: list[Passage],
    *,
    limit: int = 5,
    mode: RetrievalMode = "hybrid_late_interaction",
    embedding_backend: EmbeddingBackend | None = None,
    reranker: FeatureBasedReranker | None = None,
    skip_neural_when_confident: bool = False,
) -> RetrievalOutcome:
    empty = RetrievalTrace("none", False, 0.0, "empty_query_or_corpus")
    if not query.strip() or not passages:
        return RetrievalOutcome([], empty)
    embedder = embedding_backend or default_embedding()
    query_tokens = tokenize(query)
    bm25_scores = BM25Index([tokenize(f"{item.title} {item.text}") for item in passages]).scores(
        query_tokens
    )
    documents = [f"{item.title} {item.text}" for item in passages]
    encoded = embedder.embed_many([query, *documents])
    query_vector, doc_vectors = encoded[0], encoded[1:]
    vector_scores = [cosine(query_vector, document) for document in doc_vectors]
    keyword = keyword_scores(query, passages)
    by_id = {item.passage_id: item for item in passages}
    score_map: dict[str, float]
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    gap = vector_confidence_gap(vector_scores)
    trace = RetrievalTrace("none", False, gap, mode)

    def assign_ranks(name: str, ranking: list[str]) -> None:
        for rank, passage_id in enumerate(ranking, start=1):
            ranks[passage_id][name] = rank

    keyword_ids = _ranked_ids(keyword, passages)
    bm25_ids = _ranked_ids(bm25_scores, passages)
    vector_ids = _ranked_ids(vector_scores, passages)
    assign_ranks("keyword", keyword_ids)
    assign_ranks("bm25", bm25_ids)
    assign_ranks("vector", vector_ids)

    if mode == "keyword":
        score_map = {passages[i].passage_id: keyword[i] for i in range(len(passages))}
    elif mode == "bm25":
        score_map = {passages[i].passage_id: bm25_scores[i] for i in range(len(passages))}
    elif mode == "vector":
        score_map = {passages[i].passage_id: vector_scores[i] for i in range(len(passages))}
    else:
        score_map = rrf_fuse(bm25_ids, vector_ids)
        assign_ranks(
            "rrf",
            [pid for pid, _ in sorted(score_map.items(), key=lambda item: (-item[1], item[0]))],
        )
        if mode == "hybrid_rerank":
            encoder = reranker or FeatureBasedReranker()
            reranked: dict[str, float] = {}
            for passage_id, fused in score_map.items():
                passage = by_id[passage_id]
                index = passages.index(passage)
                reranked[passage_id] = (
                    encoder.score_pair(
                        query,
                        documents[index],
                        bm25_scores[index],
                        vector_scores[index],
                    )
                    + fused
                )
            score_map = reranked
            trace = RetrievalTrace("feature_based", False, gap, "feature_rerank")
            assign_ranks(
                "rerank",
                [pid for pid, _ in sorted(score_map.items(), key=lambda item: (-item[1], item[0]))],
            )
        elif mode == "hybrid_late_interaction":
            skip = skip_neural_when_confident and gap >= _NEURAL_CONFIDENCE_GAP
            if skip:
                trace = RetrievalTrace("rrf_selective", True, gap, "vector_gap_skip_neural")
            else:
                neural = MiniLMLateInteractionReranker().score_documents(
                    query, documents, embedder, query_vector=query_vector
                )
                reranked = {}
                for passage_id, fused in score_map.items():
                    index = passages.index(by_id[passage_id])
                    reranked[passage_id] = (
                        0.62 * neural[index] + 0.28 * vector_scores[index] + 0.10 * fused
                    )
                score_map = reranked
                trace = RetrievalTrace("late_interaction_maxsim", False, gap, "minilm_maxsim")
                assign_ranks(
                    "late_interaction",
                    [
                        pid
                        for pid, _ in sorted(
                            score_map.items(), key=lambda item: (-item[1], item[0])
                        )
                    ],
                )
        elif mode == "hybrid_cross_encoder":
            skip = skip_neural_when_confident and gap >= _NEURAL_CONFIDENCE_GAP
            if skip:
                trace = RetrievalTrace("rrf_selective", True, gap, "vector_gap_skip_cross_encoder")
            else:
                from .cross_encoder import default_cross_encoder

                candidate_ids = list(score_map.keys()) or [item.passage_id for item in passages]
                texts = [documents[passages.index(by_id[pid])] for pid in candidate_ids]
                ce_scores = default_cross_encoder().score_pairs(query, texts)
                reranked = {pid: score for pid, score in zip(candidate_ids, ce_scores, strict=True)}
                score_map = reranked
                trace = RetrievalTrace("ms_marco_minilm_l6_onnx", False, gap, "cross_encoder")
                assign_ranks(
                    "cross_encoder",
                    [
                        pid
                        for pid, _ in sorted(
                            score_map.items(), key=lambda item: (-item[1], item[0])
                        )
                    ],
                )

    ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    hits: list[RankedHit] = []
    drop_nonpositive = mode in {"keyword", "bm25"}
    for passage_id, score in ordered:
        if drop_nonpositive and score <= 0:
            continue
        hits.append(RankedHit(by_id[passage_id], score, dict(ranks.get(passage_id, {}))))
        if len(hits) >= limit:
            break
    return RetrievalOutcome(hits, trace)


EVAL_CORPUS: list[Passage] = [
    Passage(
        "p-planning",
        "src-planning",
        "section",
        "Agent Planning Notes",
        "HTN planners recursively rewrite compound goals until only primitive operators remain.",
    ),
    Passage(
        "p-arxiv",
        "src-arxiv",
        "section",
        "ReAct Preprint Note",
        "The ReAct preprint shows interleaving thought traces with tool actions on knowledge tasks.",
    ),
    Passage(
        "p-plot",
        "src-plot",
        "section",
        "Series Plot Notes",
        "A deterministic SVG polyline reports mean, standard deviation, and slope of a numeric series.",
    ),
    Passage(
        "p-citation",
        "src-citation",
        "section",
        "Citation Alignment",
        "Extractive abstracts must keep every claim aligned to a cited passage_id.",
    ),
    Passage(
        "p-quota",
        "src-quota",
        "section",
        "Quota Ledger",
        "Daily token quotas reset at 00:00 UTC and unused credits never roll over.",
    ),
    Passage(
        "p-untrusted",
        "src-untrusted",
        "section",
        "Observation Boundary",
        "Retrieved passages are wrapped as untrusted data and must never become system policy.",
    ),
    Passage(
        "p-bm25",
        "src-bm25",
        "section",
        "Lexical Ranking Primer",
        "Okapi BM25 ranks documents by term frequency saturated with k1 and length normalization b.",
    ),
    Passage(
        "p-rrf",
        "src-rrf",
        "section",
        "Fusion Notes",
        "Reciprocal Rank Fusion adds 1/(k+rank) across lexical and dense result lists.",
    ),
    Passage(
        "p-license",
        "src-license",
        "section",
        "Attribution Note",
        "Third-party model cards stay Apache-2.0; this repository redistributes no weight files.",
    ),
]


DISTRACTOR_CORPUS: list[Passage] = [
    Passage(
        "p-rainfall-gauge",
        "src-weather",
        "section",
        "Rainfall Ledger",
        "The rainfall ledger lists millimetres collected at the upland gauge.",
    ),
    Passage(
        "p-polyline-color",
        "src-svg",
        "section",
        "Stroke Log",
        "The chart stroke used a blue hex value and was not a retrieval citation.",
    ),
    Passage(
        "p-thought-only",
        "src-trace",
        "section",
        "Trace Notes",
        "A thought trace without a tool call is not a published preprint abstract.",
    ),
]


HARD_QUERIES: list[dict[str, Any]] = [
    {"query": "vendor checkpoints remain unbundled", "relevant": ["p-license"], "family": "zero_overlap"},
    {"query": "yao style interleaved planner", "relevant": ["p-arxiv"], "family": "zero_overlap"},
    {
        "query": "leftover allowance each calendar day",
        "relevant": ["p-quota"],
        "family": "zero_overlap",
    },
    {"query": "oss notice omits binaries", "relevant": ["p-license"], "family": "zero_overlap"},
    {"query": "nested objectives become leaf actions", "relevant": ["p-planning"], "family": "zero_overlap"},
    {"query": "quote grounded summary rule", "relevant": ["p-citation"], "family": "zero_overlap"},
]


EVAL_QUERIES: list[dict[str, Any]] = [
    {"query": "HTN planners recursively rewrite", "relevant": ["p-planning"], "family": "lexical"},
    {
        "query": "Which notes describe rewriting compound goals as primitive operators?",
        "relevant": ["p-planning"],
        "family": "semantic",
    },
    {
        "query": "interleaving thought traces with tool actions",
        "relevant": ["p-arxiv"],
        "family": "lexical",
    },
    {
        "query": "Which note describes ReAct thought traces and tools?",
        "relevant": ["p-arxiv"],
        "family": "semantic",
    },
    {"query": "deterministic SVG polyline", "relevant": ["p-plot"], "family": "lexical"},
    {
        "query": "Which notes mention mean and slope of a numeric series?",
        "relevant": ["p-plot"],
        "family": "semantic",
    },
    {"query": "cited passage_id", "relevant": ["p-citation"], "family": "lexical"},
    {"query": "extractive abstracts aligned to citations", "relevant": ["p-citation"], "family": "semantic"},
    {"query": "quotas reset at 00:00 UTC", "relevant": ["p-quota"], "family": "lexical"},
    {
        "query": "Which ledger says unused credits never roll over?",
        "relevant": ["p-quota"],
        "family": "semantic",
    },
    {
        "query": "untrusted data must never become system policy",
        "relevant": ["p-untrusted"],
        "family": "lexical",
    },
    {
        "query": "Which passage says retrieved text must never become policy?",
        "relevant": ["p-untrusted"],
        "family": "semantic",
    },
    {"query": "Okapi BM25 k1 length normalization", "relevant": ["p-bm25"], "family": "lexical"},
    {
        "query": "Which primer explains term frequency saturation and length normalization?",
        "relevant": ["p-bm25"],
        "family": "semantic",
    },
    {"query": "Reciprocal Rank Fusion 1/(k+rank)", "relevant": ["p-rrf"], "family": "lexical"},
    {
        "query": "Which notes describe adding 1/(k+rank) across result lists?",
        "relevant": ["p-rrf"],
        "family": "semantic",
    },
    {"query": "quote grounded summary rule", "relevant": ["p-citation"], "family": "zero_overlap"},
    {"query": "yao style interleaved planner", "relevant": ["p-arxiv"], "family": "zero_overlap"},
    {
        "query": "leftover allowance each calendar day",
        "relevant": ["p-quota"],
        "family": "zero_overlap",
    },
]


def evaluate_modes(
    queries: list[dict[str, Any]] | None = None,
    passages: list[Passage] | None = None,
    k: int = 3,
    embedding_backend: EmbeddingBackend | None = None,
) -> dict[str, dict[str, float]]:
    corpus = passages or EVAL_CORPUS
    items = queries or EVAL_QUERIES
    modes: list[RetrievalMode] = [
        "keyword",
        "bm25",
        "vector",
        "hybrid",
        "hybrid_rerank",
        "hybrid_late_interaction",
        "hybrid_cross_encoder",
    ]
    summary: dict[str, dict[str, float]] = {}
    for mode in modes:
        hits_at_k = 0
        reciprocal = 0.0
        for item in items:
            ranking = [
                hit.passage.passage_id
                for hit in retrieve(
                    item["query"],
                    corpus,
                    limit=k,
                    mode=mode,
                    embedding_backend=embedding_backend,
                )
            ]
            relevant = set(item["relevant"])
            if relevant & set(ranking):
                hits_at_k += 1
            rank = next(
                (index for index, pid in enumerate(ranking, start=1) if pid in relevant), None
            )
            reciprocal += 0.0 if rank is None else 1.0 / rank
        n = max(1, len(items))
        summary[mode] = {
            "recall_at_k": round(hits_at_k / n, 4),
            "mrr": round(reciprocal / n, 4),
            "queries": float(n),
        }
    return summary


def evaluate_zero_overlap(
    queries: list[dict[str, Any]] | None = None,
    passages: list[Passage] | None = None,
    k: int = 3,
    embedding_backend: EmbeddingBackend | None = None,
) -> dict[str, dict[str, float]]:
    items = [item for item in (queries or EVAL_QUERIES) if item.get("family") == "zero_overlap"]
    corpus = passages or EVAL_CORPUS
    for item in items:
        passage = next(row for row in corpus if row.passage_id == item["relevant"][0])
        if lexical_overlap(item["query"], f"{passage.title} {passage.text}") != 0:
            raise ValueError(f"zero_overlap query still shares tokens: {item['query']}")
    return evaluate_modes(items, corpus, k=k, embedding_backend=embedding_backend)


def evaluate_hard_retrieval(
    k: int = 3,
    embedding_backend: EmbeddingBackend | None = None,
) -> dict[str, dict[str, float]]:
    """Zero-overlap paraphrases against the corpus plus distractor passages."""
    corpus = EVAL_CORPUS + DISTRACTOR_CORPUS
    for item in HARD_QUERIES:
        passage = next(row for row in corpus if row.passage_id == item["relevant"][0])
        if lexical_overlap(item["query"], f"{passage.title} {passage.text}") != 0:
            raise ValueError(f"hard query still shares tokens: {item['query']}")
    return evaluate_modes(HARD_QUERIES, corpus, k=k, embedding_backend=embedding_backend)

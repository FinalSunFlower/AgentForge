from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from services.agent_runtime.app.eval_harness import (
    hard_retrieval_report,
    memory_eval_report,
    needle_survives_compression,
    retrieval_mode_report,
    run_embedding_router_eval,
    run_embedding_supervisor_eval,
    run_hard_embedding_eval,
    run_hard_keyword_eval,
    run_keyword_router_eval,
    run_supervisor_eval,
    zero_overlap_report,
)

router = APIRouter(tags=["evals"])
SNAPSHOT_PATH = Path(__file__).resolve().parents[4] / "data" / "evals_snapshot.json"


@lru_cache(maxsize=1)
def evals_summary_payload() -> dict[str, Any]:
    """Pure computation over the checked-in corpus. No live model call."""
    zero = zero_overlap_report()
    vector = zero.get("vector") or {}
    keyword_router = run_keyword_router_eval()[1]
    embedding_router = run_embedding_router_eval()[1]
    return {
        "retrieval": retrieval_mode_report(),
        "zero_overlap": {
            **zero,
            "minilm_recall_at_3": vector.get("recall_at_k"),
        },
        "needle": {
            "survives_compression": needle_survives_compression(),
            "scope": "structured_extractive_facts",
        },
        "memory": memory_eval_report(),
        "routing": {
            "keyword": keyword_router,
            "embedding": embedding_router,
            "supervisor_keyword": run_supervisor_eval()[1],
            "supervisor_embedding": run_embedding_supervisor_eval()[1],
            "hard_keyword": run_hard_keyword_eval()[1],
            "hard_embedding": run_hard_embedding_eval()[1],
        },
        "hard_retrieval": hard_retrieval_report(),
        "disclaimer": (
            "Retrieval and needle numbers are computed from the checked-in corpus and "
            "extractive compressor. Agent/supervisor task_success_rate values are "
            "keyword-router and MiniLM-embedding-router baselines, not live-LLM routing. "
            "hybrid_cross_encoder is the off-the-shelf MS MARCO MiniLM-L-6 ONNX model "
            "(no local training). The hard suites are trap/paraphrase/multi-step tasks "
            "and are expected to score lower."
        ),
    }


def load_evals_snapshot() -> dict[str, Any] | None:
    if not SNAPSHOT_PATH.exists():
        return None
    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


@router.get("/v1/evals/summary")
async def evals_summary(live: bool = Query(default=False)) -> dict[str, Any]:
    if not live:
        snapshot = load_evals_snapshot()
        if snapshot is not None:
            return {**snapshot, "source": "snapshot"}
    return {**evals_summary_payload(), "source": "live"}

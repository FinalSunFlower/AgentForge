from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
_ONNX_NAME = "model_quantized.onnx"
_TOKENIZER_NAME = "tokenizer.json"
_MIRRORS = (
    "https://huggingface.co/{model}/resolve/main/{path}",
    "https://hf-mirror.com/{model}/resolve/main/{path}",
)
_session: Any = None
_tokenizer: Any = None


def _cache_dir() -> Path:
    override = os.environ.get("FASTEMBED_CACHE_PATH")
    root = Path(override) if override else Path.home() / ".cache" / "fastembed"
    path = root / "ms-marco-MiniLM-L-6-v2-onnx"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, destination: Path) -> None:
    import httpx

    tmp = destination.with_suffix(".part")
    with (
        httpx.Client(timeout=120.0, follow_redirects=True) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 64):
                handle.write(chunk)
    tmp.replace(destination)


def _ensure_file(relative: str, *, min_bytes: int) -> Path:
    destination = _cache_dir() / Path(relative).name
    if destination.exists() and destination.stat().st_size > min_bytes:
        return destination
    errors: list[str] = []
    for template in _MIRRORS:
        url = template.format(model=_MODEL_NAME, path=relative)
        try:
            _download(url, destination)
            if destination.exists() and destination.stat().st_size > min_bytes:
                return destination
        except Exception as exc:  # noqa: BLE001 — try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError("cross_encoder_download_failed: " + " | ".join(errors))


def _load() -> tuple[Any, Any]:
    global _session, _tokenizer
    if _session is not None and _tokenizer is not None:
        return _session, _tokenizer
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer_path = _ensure_file(_TOKENIZER_NAME, min_bytes=10_000)
    onnx_path = _ensure_file(f"onnx/{_ONNX_NAME}", min_bytes=1_000_000)
    _tokenizer = Tokenizer.from_file(str(tokenizer_path))
    _tokenizer.enable_truncation(max_length=256)
    _tokenizer.enable_padding(length=256)
    _session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return _session, _tokenizer


class MiniLMCrossEncoder:
    """Off-the-shelf MS MARCO MiniLM-L-6 cross-encoder (quantized ONNX, CPU).

    Pretrained weights, no local fine-tune. Not bge-reranker. Eval column only;
    production retrieval stays MiniLM MaxSim.
    """

    name = "ms-marco-MiniLM-L-6-v2-onnx-int8"

    def __init__(self) -> None:
        self._pair_cache: dict[tuple[str, tuple[str, ...]], list[float]] = {}

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        cache_key = (query, tuple(documents))
        cached = self._pair_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        session, tokenizer = _load()
        encodings = [tokenizer.encode(query, document[:800]) for document in documents]
        input_ids = np.stack([np.array(item.ids, dtype=np.int64) for item in encodings])
        attention = np.stack([np.array(item.attention_mask, dtype=np.int64) for item in encodings])
        feeds: dict[str, np.ndarray] = {}
        for onnx_input in session.get_inputs():
            if onnx_input.name in {"input_ids", "inputs_ids"}:
                feeds[onnx_input.name] = input_ids
            elif onnx_input.name == "attention_mask":
                feeds[onnx_input.name] = attention
            elif onnx_input.name in {"token_type_ids", "token_type_id"}:
                type_ids = np.stack([np.array(item.type_ids, dtype=np.int64) for item in encodings])
                feeds[onnx_input.name] = type_ids
        logits = session.run(None, feeds)[0]
        flat = np.array(logits).reshape(len(documents), -1)
        if flat.shape[1] == 1:
            scores = [float(row[0]) for row in flat]
        else:
            scores = [float(row[-1]) for row in flat]
        self._pair_cache[cache_key] = scores
        return list(scores)


_DEFAULT: MiniLMCrossEncoder | None = None


def default_cross_encoder() -> MiniLMCrossEncoder:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MiniLMCrossEncoder()
    return _DEFAULT

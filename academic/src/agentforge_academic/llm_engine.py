from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class EngineResult:
    text: str
    tokens: int
    wall_ms: float


class LanguageEngine(Protocol):
    def complete(self, system: str, user: str) -> EngineResult: ...


class UsageMeter:
    tokens: int = 0
    wall_ms: float = 0.0

    def add(self, tokens: int, wall_ms: float) -> None:
        self.tokens += tokens
        self.wall_ms += wall_ms

    def take(self) -> tuple[int, float]:
        tokens, wall_ms = self.tokens, self.wall_ms
        self.tokens = 0
        self.wall_ms = 0.0
        return tokens, wall_ms


METER = UsageMeter()
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_FALLBACKS = ("Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct")


class FakeEngine:
    """Deterministic stand-in so tests never download weights."""

    def complete(self, system: str, user: str) -> EngineResult:
        del system
        reward = -0.04
        if "jump" in user or "grasp" in user or "take" in user:
            reward = 0.4
        payload = {
            "reward": reward,
            "done": 0.2,
            "confidence": 0.55,
            "next": {"position": 0, "progress": 0},
        }
        text = json.dumps(payload)
        result = EngineResult(text, tokens=12, wall_ms=0.5)
        METER.add(result.tokens, result.wall_ms)
        return result


class HuggingFaceEngine:
    """Single frozen instruct model shared by every prompted expert. No training."""

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = 64,
        device: str | None = None,
        fallback_model_ids: list[str] | None = None,
    ) -> None:
        self.candidates = [model_id, *(fallback_model_ids or [])]
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Prompted experts need the llm extra: pip install -e '.[llm]' "
                "and a CUDA torch build from https://pytorch.org"
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        errors: list[str] = []
        for model_id in self.candidates:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, dtype=dtype, **load_kwargs
                    )
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=dtype, **load_kwargs
                    )
                self._model = model.to(device)
                self._model.eval()
                for parameter in self._model.parameters():
                    parameter.requires_grad_(False)
                self.model_id = model_id
                self.device = device
                print(f"loaded_frozen_model:{model_id} device={device}", flush=True)
                return
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(f"{model_id}: {exc}")
                self._model = None
                self._tokenizer = None
        raise RuntimeError("frozen_model_load_failed: " + " | ".join(errors))

    def complete(self, system: str, user: str) -> EngineResult:
        self._load()
        assert self._model is not None and self._tokenizer is not None
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{system}\n\n{user}\n"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        wall_ms = (time.perf_counter() - started) * 1000
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens = int(inputs["input_ids"].shape[1] + new_tokens.shape[0])
        result = EngineResult(text, tokens, wall_ms)
        METER.add(result.tokens, result.wall_ms)
        return result


def parse_prediction_json(text: str) -> dict[str, Any] | None:
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    match = _JSON_BLOCK.search(candidate)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def get_engine(config: dict[str, Any]) -> LanguageEngine:
    kind = str(config.get("engine", "huggingface"))
    if kind == "fake":
        return FakeEngine()
    device = config.get("device")
    fallbacks = [str(item) for item in config.get("fallback_model_ids") or DEFAULT_FALLBACKS]
    return HuggingFaceEngine(
        str(config.get("model_id", DEFAULT_MODEL_ID)),
        max_new_tokens=int(config.get("max_new_tokens", 64)),
        device=str(device) if device else None,
        fallback_model_ids=fallbacks,
    )

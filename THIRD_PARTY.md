# Third-party notices

This repository's original source is MIT. It does **not** vendor model
weights, Hugging Face snapshots, npm `node_modules`, or third-party paper
PDFs. Dependencies keep their own licenses.

## Pretrained models (downloaded at runtime, not checked in)

| Artifact | Source | License | How this repo uses it |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` ONNX | Qdrant FastEmbed tarball | Apache-2.0 | Product embeddings / MaxSim |
| `Xenova/ms-marco-MiniLM-L-6-v2` quantized ONNX | Hugging Face (Xenova ONNX export of the Sentence-Transformers MS MARCO MiniLM cross-encoder) | Apache-2.0 | Eval-only rerank column |
| Optional frozen instruct LMs (`HuggingFaceTB/SmolLM2-*`, `Qwen/Qwen2.5-*`) | Hugging Face, only if you install `academic[llm]` | See each model card | Optional academic smoke; not the main result |

MS MARCO **passages** are not redistributed here. The cross-encoder is an
off-the-shelf checkpoint; this project does not train or fine-tune it.

## Literature (cited, not copied)

Academic baselines named ReAct / RAP / PreAct are **original tabular
controllers on ToolWorld-v1 / Household-v1**. They are not reproductions of
the official ALFWorld, ScienceWorld, or τ-bench releases, and they are not
copies of the papers' code or figures. Citations live in
`academic/README.md`.

Figures under `figures/` and `academic/figures/` are original to this
project. `figures/architecture.jpg` is an original diagram of this runtime,
not a vendor or third-party screenshot.

## Runtime and console dependencies

Python packages are declared in `pyproject.toml` and
`academic/pyproject.toml` (FastAPI, SQLAlchemy, NumPy, Matplotlib, and
others). The Next.js console lockfile is `apps/web/package-lock.json`;
install with `npm ci`. Those trees include MIT, Apache-2.0, BSD, ISC, and
some optional native LGPL components of the Next.js toolchain. We ship the
lockfile, not `node_modules`.

The architecture page may load Mermaid from jsDelivr at browse time (MIT).
That is a CDN fetch, not a vendored copy.

# Zakirulla Khan Ghouri
**ML Systems & AI Engineer · VIT Chennai · Chennai, India**

I build **production ML systems** — not just notebooks. My focus is the gap between a trained model and something that actually runs fast under load: inference optimization, serving architecture, RAG pipelines, and LLM fine-tuning.

Interested in ML systems, inference engineering, GenAI infrastructure, and production-oriented AI deployment.

---

## ⚙ Engineering Interests along with Model Training

- Model serving & inference optimization (ONNX, vectorized batching, low latency)
- LLM fine-tuning & alignment (LoRA/QLoRA, SFT/DPO)
- RAG systems & retrieval evaluation
- GenAI infrastructure
- Production deployment & MLOps tradeoffs

---

## 🔧 Projects

### [BA Lead Engine](https://github.com/mystic0137/BA_Lead_Engine) — `complete`

> **5–7× batch throughput · 82.3% ROC-AUC · 160+ tests · offline-capable inference**

Production-oriented lead prioritization and GenAI outreach system that scores **50K airline booking records** with XGBoost → ONNX Runtime → FastAPI, then ranks every lead by expected value for sales.

- Cut batch inference latency **5–7× (106ms → 15ms at 1k leads)** with a fully vectorized, column-oriented NumPy pipeline replacing the row-by-row path.
- Designed a singleton `MLState` architecture so heavy runtime objects (ONNX session, embedding model, ChromaDB) initialize once and are shared across requests, with lazy RAG loading off the prediction hot path.
- Built a Retrieval-Augmented Generation email pipeline (ChromaDB + Groq / Llama-4-Scout) grounded in 9 BA policy documents — **80% prompt-context reduction (20k → 4k tokens)** with multi-provider failover (Groq → Together AI → Ollama).
- Added RLHF-ready feedback logging that exports **SFT/DPO JSONL datasets** with contradiction filtering, plus a thread-safe ONNX engine and **160+ mocked tests** running in under 1s.
- Ships via **multi-stage Docker** with offline-capable inference — XGBoost trained at build time and embeddings cached in-image, so scoring and RAG retrieval need no network; only the final LLM generation call uses Groq/Together (or a local Ollama fallback).

`XGBoost` `ONNX Runtime` `FastAPI` `Docker` `ChromaDB` `Pydantic V2` `Groq`

---

### [Prompt-to-PEFT](https://github.com/mystic0137/LLM-finetuning) — `complete`

> **Hypothesis: prompt failure modes should guide fine-tuning targets — not overall accuracy. Validated.**

Research-oriented LLM evaluation + PEFT project investigating how prompt structure, instruction framing, and pretraining priors shape downstream classification — and whether targeted fine-tuning can fix what prompting cannot. Built on a **cache-first, reproducible pipeline** (Parquet prediction store keyed by model/prompt/dataset hash) so all statistics rerun without repeating GPU inference.

**Phase 1 — Prompt engineering baseline** (zero/few-shot ablations across Llama-3.2-1B, Qwen2.5-1.5B, Gemma-3-1B on 1,909 PubMed abstracts):
- Best zero-shot accuracy: **45.84%**; explicit task framing alone gave a **+27 point gain** with no semantic change.
- Diagnosed persistent **Meta-analysis over-prediction (48% of predictions vs. 19% true prevalence)** and traced it to pretraining priors, not prompt structure — confirmed with bootstrap CIs, McNemar + FDR, and a 30-dataset sensitivity analysis.

**Phase 2 — Statistically-gated PEFT** (QLoRA on Llama, targeting the diagnosed failure class):
- Held-out accuracy **43.4% → 71.4% (+28 pts)** on 8,169 unseen papers; macro-F1 **0.41 → 0.72**, McNemar p ≈ 0.
- Automated pipeline that **promotes an adapter only when McNemar confirms a significant gain with no macro-F1 regression** — closing the diagnose → tune → verify loop.

`PyTorch` `LoRA / QLoRA` `PEFT` `Hugging Face Transformers` `Llama-3.2-1B` `Qwen` `Gemma` `PubMed`

---

## 🛠 Core Stack

**ML & Modeling:** `Python` · `PyTorch` `Hugging Face` `Transformers` · `XGBoost` · `scikit-learn` · `NumPy` · `pandas`

**LLM / GenAI:** `LoRA / QLoRA / PEFT` · `RAG Pipelines` · `ChromaDB` · `Sentence-Transformers` · `Groq / Llama`

**Serving & Infra:** `ONNX Runtime` · `FastAPI` · `Docker` · `Pydantic V2` · `Vectorized Inference`

**Testing & Tooling:** `pytest` · `Streamlit` · `Git`

---

## 📬 Links

- 🌐 Website: https://mystic0137.github.io
- ✉️ Email: zakir.ghouri.dev@gmail.com
- 💻 GitHub: https://github.com/mystic0137

---

*Open to ML/AI Engineer / ML Systems Engineer roles · Remote or relocation*

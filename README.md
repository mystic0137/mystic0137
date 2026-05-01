# Zakirulla Khan Ghouri

**AI Engineer · VIT Chennai · Chennai, India**

I build **production ML systems** — not notebooks. My focus is the gap between a trained model and something that actually runs fast under load: inference optimization, serving architecture, RAG pipelines, and LLM fine-tuning.

---

## 🔧 What I'm working on

### [BA Lead Engine](https://github.com/mystic0137/BA_Lead_Engine) — `complete`
> **36.2× throughput improvement**

End-to-end booking prediction system. XGBoost + Random Forest → ONNX → FastAPI with vectorized batch inference. RAG pipeline for policy email generation (ChromaDB + Groq/Llama-4-Scout). Threshold-optimized via Youden's J.

`XGBoost` `ONNX Runtime` `FastAPI` `ChromaDB` `Pydantic V2`

---

### [Prompt-to-PEFT](https://github.com/mystic0137/LLM-finetuning) — `in progress`

Investigating whether systematic prompt engineering can surface the exact failure modes that fine-tuning should target. 5-class medical paper classification (1909 PubMed abstracts) using Llama-3.2-1B-Instruct.

Phase 1 complete: zero-shot ablations hit 45.84% accuracy — isolated pretraining prior bias as the root cause of Meta-analysis over-prediction (48% of predictions vs. 19% true prevalence). Confirmed the failure hierarchy is model-prior-driven, not prompt-format-driven.

Phase 2 in progress: LoRA/QLoRA fine-tuning targeting Observational Studies, the hardest class (best recall 0.46 under any prompting strategy).

`PyTorch` `LoRA / QLoRA` `HuggingFace` `Llama-3.2-1B` `PubMed API`

---

## 🛠 Stack

| | |
|---|---|
| **Languages** | Python, SQL |
| **ML / AI** | PyTorch, XGBoost, Scikit-learn, PEFT / LoRA / QLoRA |
| **Serving** | ONNX Runtime, FastAPI, Pydantic V2 |
| **GenAI** | ChromaDB, Sentence-Transformers, Groq API |

---

## 📬 Links

- 🌐 [mystic0137.github.io](https://mystic0137.github.io)
- ✉️ zakir.ghouri.dev@gmail.com

---

*Open to AI Engineer roles · Remote or relocation*

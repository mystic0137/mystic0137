# Zakirulla Khan Ghouri
**ML Systems & AI Engineer · VIT Chennai · Chennai, India**

I build **production ML systems** — not just notebooks. My focus is the gap between a trained model and something that actually runs fast under load: inference optimization, serving architecture, RAG pipelines, and LLM fine-tuning.

Interested in ML systems, inference engineering, GenAI infrastructure, and production-oriented AI deployment.

---

## ⚙ Engineering Interests

- Low-latency ML inference
- GenAI infrastructure
- Vectorized batch processing
- RAG systems & evaluation
- LLM reliability & schema validation
- Production deployment tradeoffs

---

## 🔧 Projects

### [BA Lead Engine](https://github.com/mystic0137/BA_Lead_Engine) — `complete`

> **5–7× batch throughput · 160+ tests · air-gapped Docker deployment**

Production-oriented lead prioritization and GenAI email system using XGBoost → ONNX Runtime → FastAPI with vectorized column-oriented batch inference (15ms vs. 106ms row-oriented at 1k leads).

Designed a singleton `MLState` architecture so heavy runtime objects (ONNX sessions, embeddings, ChromaDB) are initialized once and shared across requests.

Built a Retrieval-Augmented Generation (RAG) email pipeline using ChromaDB and Groq/Llama-4-Scout with 5× prompt reduction (20k → 4k tokens) and multi-provider failover routing (Groq → Together AI → Ollama).

Includes RLHF-ready feedback logging generating SFT/DPO JSONL datasets and fully air-gapped Docker deployment with offline embedding support.

`XGBoost` `ONNX Runtime` `FastAPI` `Docker` `ChromaDB` `Pydantic V2` `Groq`

---

### [Prompt-to-PEFT](https://github.com/mystic0137/LLMFinetuning) — `in progress`

> **Hypothesis: prompt failure modes should guide fine-tuning targets, not overall accuracy**

Research-oriented LLM evaluation project investigating how prompt structure, instruction framing, and pretraining priors influence downstream classification behavior.

Conducted systematic zero-shot and few-shot ablations on 1,909 PubMed medical abstracts using Llama-3.2-1B-Instruct.

Phase 1 findings:
- Best zero-shot accuracy: **45.84%**
- Explicit task framing alone produced a **+27 point accuracy gain**
- Identified persistent Meta-analysis over-prediction (48% predictions vs. 19% true prevalence) and traced the failure mode to pretraining priors rather than prompt structure through controlled zero-shot and few-shot ablations

Phase 2 in progress:
- Targeted LoRA/QLoRA fine-tuning for hardest-class recovery
- GRPO implementation from scratch in PyTorch

`PyTorch` `LoRA / QLoRA` `PEFT` `Hugging Face Transformers` `Llama-3.2-1B` `PubMed`

---

## 🛠 Core Stack

`Python`
`PyTorch`
`ONNX Runtime`
`FastAPI`
`Docker`
`ChromaDB`
`Sentence-Transformers`
`LoRA / QLoRA / PEFT`
`Hugging Face Transformers`
`Pydantic V2`
`Vectorized Inference`
`RAG Pipelines`

---

## 📬 Links

- 🌐 Website: https://mystic0137.github.io
- ✉️ Email: zakir.ghouri.dev@gmail.com
- 💻 GitHub: https://github.com/mystic0137

---

*Open to ML/AI Engineer / ML Systems Engineer roles · Remote or relocation*

# NVIDIA GenAI Learning Projects

Hands-on projects exploring NVIDIA's AI/GenAI ecosystem: NIM, NeMo, TensorRT-LLM, and more.
Each project is self-contained with its own README, code, and documented use cases.

---

## Learning Roadmap

### Phase 1 — NVIDIA NIM (Inference Microservices)
> Run optimized AI models via API — no GPU required, OpenAI-compatible interface.

| Project | What it covers | Status |
|---|---|---|
| `project1-nim-llm-chat` | LLM chat with Llama 3.1 via NIM cloud API | ✅ Done |
| `project2-nim-rag-pipeline` | RAG pipeline using NIM embeddings + vector DB | ✅ Done |
| `project3-nim-vision-language` | Visual Q&A and multi-turn chat with a VLM | ✅ Done |

### Phase 2 — NeMo Guardrails
> Add programmable safety, topic control, and jailbreak protection to LLM apps.

| Project | What it covers | Status |
|---|---|---|
| `project4-multimodal-rag` | Multimodal RAG — documents + images + multi-image comparison | ✅ Done |
| `project5-nemo-guardrails` | Chat app with topic guardrails and jailbreak protection | ⏳ Upcoming |

### Phase 3 — NeMo Framework (Training & Fine-tuning)
> Train and customize LLMs and speech models at scale.

| Project | What it covers | Status |
|---|---|---|
| `project6-nemo-lora-finetuning` | Fine-tune a small LLM with LoRA/PEFT | ⏳ Upcoming |
| `project7-nemo-speech-pipeline` | ASR + TTS voice pipeline using NeMo | ⏳ Upcoming |

### Phase 4 — Ecosystem Depth
> TensorRT-LLM, Triton Inference Server, RAPIDS GPU-accelerated data science.

| Project | What it covers | Status |
|---|---|---|
| `project8-tensorrt-llm-concepts` | TensorRT-LLM optimization concepts and benchmarking | ⏳ Upcoming |
| `project9-triton-inference-server` | Serving models with Triton Inference Server | ⏳ Upcoming |

---

## Setup

Each project folder has its own README and dependencies. For all NIM-based projects you need:

1. A free API key from [build.nvidia.com](https://build.nvidia.com)
2. A `.env` file at the repo root (never committed — protected by `.gitignore`):
   ```
   NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
   ```

---

## Key Concepts Covered

- **NIM**: Containerized inference microservices — deploy anywhere, same API
- **NeMo Guardrails**: Programmable safety rails for production LLM apps  
- **NeMo Framework**: End-to-end LLM and speech model training/fine-tuning
- **TensorRT-LLM**: NVIDIA's inference optimization engine (used under the hood by NIM)
- **Triton**: Production model serving framework
- **RAG**: Retrieval-Augmented Generation — grounding LLMs in your own data

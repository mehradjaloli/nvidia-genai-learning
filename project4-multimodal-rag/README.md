# Project 4 — Multimodal RAG with NVIDIA NIM

## What we built
A single interface that combines document RAG and Vision Language Models.
The system automatically routes each query to the right engine based on what you upload.

## Architecture

```
User query
     │
     ▼
 Orchestrator  ──── 2 images? ──────────────► Multi-image Comparison (VLM)
     │
     ├─── image + documents? ──────────────► Hybrid: RAG context + VLM
     │
     ├─── image only? ─────────────────────► Vision Q&A (VLM)
     │
     └─── documents only? ─────────────────► Document RAG (LLM)
```

## Modes

| Mode | Trigger | Models used |
|---|---|---|
| **Document RAG** | Documents indexed, no image | `nv-embedqa-e5-v5` + `llama-3.1-8b-instruct` |
| **Vision Q&A** | Image uploaded, no documents | `llama-3.2-11b-vision-instruct` |
| **Hybrid** | Image + documents both present | `nv-embedqa-e5-v5` + `llama-3.2-11b-vision-instruct` |
| **Multi-image Comparison** | Two images uploaded | `llama-3.2-11b-vision-instruct` |

## Project structure
```
project4-multimodal-rag/
├── rag_engine.py      # document chunking, embedding, FAISS index, retrieval
├── vision_client.py   # image encoding, VLM calls, multi-image comparison
├── orchestrator.py    # routing logic — decides which engine(s) to use
├── app.py             # Gradio UI
└── README.md          # this file
```

## Key concept: the orchestrator pattern
Separating routing logic into its own module (`orchestrator.py`) means the
UI layer stays clean and the engines are independently testable. This is the
standard pattern for production multimodal AI systems.

## Setup
```bash
pip install openai python-dotenv faiss-cpu numpy gradio pypdf
```

## Run
```bash
cd project4-multimodal-rag
python3 app.py
```

## Things to try
1. Upload a PDF → ask questions about it (RAG mode)
2. Upload an image → ask what's in it (VLM mode)
3. Upload both a PDF and an image → ask a question that combines both (Hybrid mode)
4. Upload two images → ask the model to compare them (Multi-image Comparison mode)

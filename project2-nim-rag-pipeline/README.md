# Project 2 — RAG Pipeline with NVIDIA NIM Embeddings

## What we built
A full Retrieval-Augmented Generation (RAG) pipeline using two NVIDIA NIM models:
- **Embedding model** (`nvidia/nv-embedqa-e5-v5`) — converts text to vectors
- **LLM** (`meta/llama-3.1-8b-instruct`) — generates answers from retrieved context

## What is RAG?
RAG solves a core limitation of LLMs: they only know what they were trained on.
RAG gives the LLM access to your own documents at query time, without retraining.

```
User question
     │
     ▼
Embed question  ──►  Search vector store  ──►  Retrieve top-K chunks
                                                        │
                                                        ▼
                                          Inject chunks into LLM prompt
                                                        │
                                                        ▼
                                                Generate answer
```

## Why use NIM for embeddings?
- Same API as the LLM — one SDK, one API key, one base URL
- `nv-embedqa-e5-v5` is optimized specifically for retrieval (not just similarity)
- Swap to a self-hosted NIM container by changing `base_url` — zero code changes

## Project structure
```
project2-nim-rag-pipeline/
├── ingest.py          # embed documents and build FAISS index
├── query.py           # retrieve + generate answers
├── README.md          # this file
├── vector_store.index # generated after running ingest.py (gitignored)
└── documents.json     # generated after running ingest.py (gitignored)
```

## Setup
```bash
pip install openai python-dotenv faiss-cpu numpy
```

## Run

**Step 1 — Ingest documents (run once):**
```bash
cd project2-nim-rag-pipeline
python3 ingest.py
```

**Step 2 — Query the knowledge base:**
```bash
python3 query.py
```

Try asking:
- "What is NVIDIA NIM?"
- "How does RAG work?"
- "What is the difference between NIM and NeMo?"

## Key concept: `input_type` parameter
NIM's embedding model distinguishes between embedding a *passage* (document) and a *query* (question).
This improves retrieval accuracy — always use `input_type="passage"` for documents and `input_type="query"` for questions.

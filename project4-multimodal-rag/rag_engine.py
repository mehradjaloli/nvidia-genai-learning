"""
RAG Engine (Document retrieval)
---------------------------------
Handles document ingestion, embedding, FAISS indexing, and retrieval.
Used by the orchestrator when the query requires document knowledge.
"""

import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
VECTOR_DIM = 1024
TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

INDEX_FILE = os.path.join(os.path.dirname(__file__), "vector_store.index")
DOCS_FILE = os.path.join(os.path.dirname(__file__), "documents.json")


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def embed_passages(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": "passage"},
    )
    vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


def embed_query(question: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
        encoding_format="float",
        extra_body={"input_type": "query"},
    )
    vec = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def load_index() -> tuple[faiss.Index, list[str]]:
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE) as f:
            documents = json.load(f)
    else:
        index = faiss.IndexFlatIP(VECTOR_DIM)
        documents = []
    return index, documents


def save_index(index: faiss.Index, documents: list[str]):
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "w") as f:
        json.dump(documents, f, indent=2)


def ingest_text(raw_text: str) -> int:
    chunks = chunk_text(raw_text)
    if not chunks:
        return 0
    index, documents = load_index()
    new_chunks = [c for c in chunks if c not in documents]
    if not new_chunks:
        return 0
    vectors = embed_passages(new_chunks)
    index.add(vectors)
    documents.extend(new_chunks)
    save_index(index, documents)
    return len(new_chunks)


def retrieve(question: str) -> list[str]:
    index, documents = load_index()
    if index.ntotal == 0:
        return []
    query_vec = embed_query(question)
    _, indices = index.search(query_vec, min(TOP_K, index.ntotal))
    return [documents[i] for i in indices[0] if i < len(documents)]


def get_index_stats() -> dict:
    index, documents = load_index()
    return {"total_chunks": index.ntotal}


def has_documents() -> bool:
    index, _ = load_index()
    return index.ntotal > 0

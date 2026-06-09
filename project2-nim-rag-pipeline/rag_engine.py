"""
RAG Engine
-----------
Core logic for the RAG pipeline: chunking, embedding, indexing, and retrieval.
Kept separate from the UI so it can be reused with any frontend (Gradio, FastAPI, etc.)
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
LLM_MODEL = "meta/llama-3.1-8b-instruct"
VECTOR_DIM = 1024
TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

INDEX_FILE = os.path.join(os.path.dirname(__file__), "vector_store.index")
DOCS_FILE = os.path.join(os.path.dirname(__file__), "documents.json")


# --- Chunking ---

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]  # drop tiny trailing chunks


# --- Embedding ---

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


# --- Index management ---

def load_index() -> tuple[faiss.Index, list[str]]:
    """Load existing index and documents, or create empty ones."""
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


def ingest_text(raw_text: str, source_name: str = "") -> int:
    """
    Chunk, embed, and add text to the existing index incrementally.
    Returns the number of new chunks added.
    """
    chunks = chunk_text(raw_text)
    if not chunks:
        return 0

    index, documents = load_index()

    # Deduplicate: skip chunks already in the index
    new_chunks = [c for c in chunks if c not in documents]
    if not new_chunks:
        return 0

    vectors = embed_passages(new_chunks)
    index.add(vectors)
    documents.extend(new_chunks)
    save_index(index, documents)

    return len(new_chunks)


def get_index_stats() -> dict:
    index, documents = load_index()
    return {"total_chunks": index.ntotal, "total_documents": len(documents)}


# --- Retrieval & generation ---

def retrieve(question: str) -> list[str]:
    index, documents = load_index()
    if index.ntotal == 0:
        return []
    query_vec = embed_query(question)
    _, indices = index.search(query_vec, min(TOP_K, index.ntotal))
    return [documents[i] for i in indices[0] if i < len(documents)]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    if not context_chunks:
        return "No documents have been ingested yet. Please upload some documents first."

    context = "\n\n".join(f"- {chunk}" for chunk in context_chunks)
    prompt = f"""Answer the question using only the context provided below.
If the answer is not in the context, say "I don't have that information in the uploaded documents."

Context:
{context}

Question: {question}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content


def query(question: str) -> tuple[str, list[str]]:
    """Full RAG query: retrieve + generate. Returns (answer, source_chunks)."""
    chunks = retrieve(question)
    answer = generate_answer(question, chunks)
    return answer, chunks

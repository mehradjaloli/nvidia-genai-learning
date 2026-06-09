"""
RAG Pipeline Demo — Gradio UI
------------------------------
Drag and drop documents (TXT or PDF) to incrementally add them to the
vector index. Then chat with the knowledge base in the right panel.
"""

import os
import gradio as gr
from rag_engine import ingest_text, query, get_index_stats

try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


def read_file(file_path: str, ext: str) -> str:
    """Extract text from a TXT or PDF file."""
    if ext == "pdf":
        if not PDF_SUPPORT:
            return ""
        reader = pypdf.PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def handle_upload(files) -> str:
    if not files:
        return "No files uploaded."

    results = []
    for file in files:
        # Use original filename to get extension; file.name is a temp path
        original_name = getattr(file, "orig_name", None) or os.path.basename(file.name)
        filename = original_name
        ext = filename.lower().split(".")[-1]

        if ext == "pdf" and not PDF_SUPPORT:
            results.append(f"⚠️  {filename} — PDF support requires: pip install pypdf")
            continue
        if ext not in ("txt", "pdf", "md"):
            results.append(f"⚠️  {filename} — unsupported format (use TXT, PDF, or MD)")
            continue

        raw_text = read_file(file.name, ext)
        if not raw_text.strip():
            results.append(f"⚠️  {filename} — could not extract text")
            continue

        n_chunks = ingest_text(raw_text, source_name=filename)
        if n_chunks == 0:
            results.append(f"⏭️  {filename} — already indexed, no new chunks added")
        else:
            results.append(f"✅  {filename} — {n_chunks} new chunks added to index")

    stats = get_index_stats()
    summary = f"\n📦 Index total: {stats['total_chunks']} chunks"
    return "\n".join(results) + summary


def handle_chat(message: str, history: list) -> tuple[str, list]:
    if not message.strip():
        return "", history

    answer, sources = query(message)

    source_text = "\n\n**Retrieved context:**\n" + "\n".join(
        f"> {i+1}. {chunk[:120]}..." for i, chunk in enumerate(sources)
    )
    full_response = answer + source_text
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})
    return "", history


# --- UI Layout ---

with gr.Blocks(title="NVIDIA NIM RAG Pipeline") as demo:
    gr.Markdown("# NVIDIA NIM RAG Pipeline")
    gr.Markdown(
        "Upload documents to build a knowledge base, then ask questions. "
        "Powered by **NVIDIA NIM** — `nv-embedqa-e5-v5` for embeddings, `llama-3.1-8b-instruct` for answers."
    )

    with gr.Row():
        # Left panel: document ingestion
        with gr.Column(scale=1):
            gr.Markdown("### Upload Documents")
            gr.Markdown("Drag and drop files below. New files are added **incrementally** — existing chunks are preserved.")
            file_input = gr.File(
                label="TXT, PDF, or MD files",
                file_types=[".txt", ".pdf", ".md"],
                file_count="multiple",
            )
            ingest_btn = gr.Button("Ingest Documents", variant="primary")
            ingest_output = gr.Textbox(
                label="Ingestion Status",
                lines=6,
                interactive=False,
                placeholder="Upload files and click 'Ingest Documents'...",
            )
            ingest_btn.click(fn=handle_upload, inputs=file_input, outputs=ingest_output)

        # Right panel: chat
        with gr.Column(scale=2):
            gr.Markdown("### Ask Questions")
            chatbot = gr.Chatbot(height=450, label="RAG Chat")
            msg_input = gr.Textbox(
                placeholder="Ask a question about your documents...",
                label="Your question",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear Chat")

            submit_btn.click(fn=handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            msg_input.submit(fn=handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_input])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

"""
Project 4: Multimodal RAG — Gradio UI
---------------------------------------
Combines document RAG and Vision Language Models in a single interface.

Modes (auto-detected by orchestrator):
  - Documents only      → RAG pipeline
  - Single image only   → VLM visual Q&A
  - Image + documents   → Hybrid (RAG context injected into VLM prompt)
  - Two images          → Multi-image comparison
"""

import os
import gradio as gr
from rag_engine import ingest_text, get_index_stats
from orchestrator import query

try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Global state
api_history: list = []
current_image_1: str | None = None
current_image_2: str | None = None


def read_file(file_path: str, ext: str) -> str:
    if ext == "pdf":
        if not PDF_SUPPORT:
            return ""
        reader = pypdf.PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def handle_document_upload(files) -> str:
    if not files:
        return "No files uploaded."
    results = []
    for file in files:
        original_name = getattr(file, "orig_name", None) or os.path.basename(file.name)
        ext = original_name.lower().split(".")[-1]
        if ext not in ("txt", "pdf", "md"):
            results.append(f"⚠️  {original_name} — unsupported format")
            continue
        raw_text = read_file(file.name, ext)
        if not raw_text.strip():
            results.append(f"⚠️  {original_name} — could not extract text")
            continue
        n = ingest_text(raw_text)
        if n == 0:
            results.append(f"⏭️  {original_name} — already indexed")
        else:
            results.append(f"✅  {original_name} — {n} chunks added")
    stats = get_index_stats()
    return "\n".join(results) + f"\n\n📦 Index total: {stats['total_chunks']} chunks"


def handle_image_1(image_path) -> str:
    global current_image_1, api_history
    current_image_1 = image_path
    api_history = []
    return _mode_label()


def handle_image_2(image_path) -> str:
    global current_image_2, api_history
    current_image_2 = image_path
    api_history = []
    return _mode_label()


def _mode_label() -> str:
    if current_image_1 and current_image_2:
        return "🔀 Mode: Multi-image Comparison"
    if current_image_1 and get_index_stats()["total_chunks"] > 0:
        return "🔀 Mode: Hybrid (Image + Documents)"
    if current_image_1:
        return "🖼️  Mode: Vision Q&A"
    if get_index_stats()["total_chunks"] > 0:
        return "📄 Mode: Document RAG"
    return "⏳ Mode: Upload documents or images to begin"


def handle_chat(message: str, chat_history: list) -> tuple[str, list, str]:
    global api_history

    if not message.strip():
        return "", chat_history, _mode_label()

    answer, mode, api_history = query(message, current_image_1, current_image_2, api_history)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": f"**[{mode}]**\n\n{answer}"})
    return "", chat_history, _mode_label()


def handle_clear_chat() -> tuple[list, str]:
    global api_history
    api_history = []
    return [], ""


def handle_clear_images() -> tuple[None, None, str]:
    global current_image_1, current_image_2, api_history
    current_image_1 = None
    current_image_2 = None
    api_history = []
    return None, None, _mode_label()


# --- UI ---

with gr.Blocks(title="Multimodal RAG — NVIDIA NIM") as demo:
    gr.Markdown("# Multimodal RAG — NVIDIA NIM")
    gr.Markdown(
        "Combine documents and images in one knowledge interface. "
        "The system auto-detects the right mode based on what you upload."
    )

    mode_indicator = gr.Textbox(
        value=_mode_label(), label="Current Mode", interactive=False, lines=1
    )

    with gr.Row():
        # Left panel: inputs
        with gr.Column(scale=1):
            with gr.Tab("Documents"):
                gr.Markdown("Upload TXT, PDF, or MD files to build the knowledge base.")
                doc_files = gr.File(
                    label="Documents",
                    file_types=[".txt", ".pdf", ".md"],
                    file_count="multiple",
                )
                ingest_btn = gr.Button("Ingest Documents", variant="primary")
                doc_status = gr.Textbox(label="Ingestion Status", lines=5, interactive=False)
                ingest_btn.click(fn=handle_document_upload, inputs=doc_files, outputs=doc_status)

            with gr.Tab("Images"):
                gr.Markdown(
                    "**Single image:** upload Image 1 only — ask questions about it.\n\n"
                    "**Compare two images:** upload both — the model will compare them."
                )
                image_1 = gr.Image(type="filepath", label="Image 1")
                image_2 = gr.Image(type="filepath", label="Image 2 (optional — for comparison)")
                clear_images_btn = gr.Button("Clear Images")

                image_1.change(fn=handle_image_1, inputs=image_1, outputs=mode_indicator)
                image_2.change(fn=handle_image_2, inputs=image_2, outputs=mode_indicator)
                clear_images_btn.click(fn=handle_clear_images, outputs=[image_1, image_2, mode_indicator])

        # Right panel: chat
        with gr.Column(scale=2):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(height=500, label="Conversation")
            msg_input = gr.Textbox(
                placeholder="Ask a question...",
                label="Your question",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_chat_btn = gr.Button("Clear Chat")

            submit_btn.click(
                fn=handle_chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, mode_indicator],
            )
            msg_input.submit(
                fn=handle_chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, mode_indicator],
            )
            clear_chat_btn.click(fn=handle_clear_chat, outputs=[chatbot, msg_input])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

"""
Multimodal Orchestrator
-------------------------
Routes each query to the appropriate engine based on what inputs are available.

Routing logic:
  - 2 images provided            → multi-image comparison (VLM)
  - 1 image + no documents       → single image Q&A (VLM)
  - 1 image + documents indexed  → hybrid: retrieve context + analyze image (VLM + RAG)
  - no image + documents indexed → document Q&A (RAG + LLM)
  - nothing                      → prompt user to upload something
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from rag_engine import retrieve, has_documents
from vision_client import ask_about_image, compare_images

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

LLM_MODEL = "meta/llama-3.1-8b-instruct"


def _rag_answer(question: str) -> tuple[str, str]:
    """Pure RAG: retrieve chunks → generate answer → return (answer, mode_label)."""
    chunks = retrieve(question)
    if not chunks:
        return "No relevant documents found. Please upload some documents first.", "RAG"

    context = "\n\n".join(f"- {chunk}" for chunk in chunks)
    prompt = f"""Answer the question using only the context below.
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
    answer = response.choices[0].message.content
    sources = "\n".join(f"> {i+1}. {c[:120]}..." for i, c in enumerate(chunks))
    return f"{answer}\n\n**Retrieved context:**\n{sources}", "RAG"


def _hybrid_answer(image_path: str, question: str, api_history: list) -> tuple[str, str]:
    """
    Hybrid: retrieve relevant document chunks AND analyze the image.
    Both are injected into the VLM prompt as context.
    """
    chunks = retrieve(question)
    doc_context = ""
    if chunks:
        doc_context = "\n\nRelevant document context:\n" + "\n".join(f"- {c}" for c in chunks)

    augmented_question = question + doc_context
    answer = ask_about_image(image_path, augmented_question, api_history)
    return answer, "Hybrid (RAG + VLM)"


def query(
    question: str,
    image_path_1: str | None,
    image_path_2: str | None,
    api_history: list,
) -> tuple[str, str, list]:
    """
    Main entry point for the orchestrator.
    Returns (answer, mode_label, updated_api_history).
    """
    if not question.strip():
        return "", "", api_history

    # --- Route ---
    if image_path_1 and image_path_2:
        # Multi-image comparison
        answer = compare_images(image_path_1, image_path_2, question)
        mode = "Multi-image Comparison (VLM)"
        api_history = []  # comparison is stateless across turns

    elif image_path_1 and has_documents():
        # Hybrid: image + documents
        answer, mode = _hybrid_answer(image_path_1, question, api_history)
        if not api_history:
            from vision_client import image_part
            api_history.append({
                "role": "user",
                "content": [image_part(image_path_1), {"type": "text", "text": question}],
            })
        else:
            api_history.append({"role": "user", "content": question})
        api_history.append({"role": "assistant", "content": answer})

    elif image_path_1:
        # Single image, no documents
        answer = ask_about_image(image_path_1, question, api_history)
        mode = "Vision (VLM)"
        if not api_history:
            from vision_client import image_part
            api_history.append({
                "role": "user",
                "content": [image_part(image_path_1), {"type": "text", "text": question}],
            })
        else:
            api_history.append({"role": "user", "content": question})
        api_history.append({"role": "assistant", "content": answer})

    elif has_documents():
        # Documents only, no image
        answer, mode = _rag_answer(question)

    else:
        answer = "Please upload documents or images to get started."
        mode = "None"

    return answer, mode, api_history

"""
Project 3: Vision Language Model Demo — Gradio UI
---------------------------------------------------
Upload an image and have a multi-turn conversation about it.
Powered by meta/llama-3.2-11b-vision-instruct via NVIDIA NIM.
"""

import gradio as gr
from vision_client import ask_about_image, describe_image

# Holds the current image path across turns
current_image_path: str | None = None
# Holds raw API message history for multi-turn context
api_history: list[dict] = []


def handle_image_upload(image_path: str) -> str:
    global current_image_path, api_history
    current_image_path = image_path
    api_history = []  # reset conversation when a new image is uploaded
    return "Image loaded. Ask a question or click 'Auto-Describe'."


def handle_auto_describe(chat_history: list) -> tuple[list, str]:
    global api_history

    if not current_image_path:
        chat_history.append({"role": "assistant", "content": "Please upload an image first."})
        return chat_history, ""

    description = describe_image(current_image_path)

    # Sync api_history so follow-up questions have context
    api_history = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _get_data_url(current_image_path)}},
                {"type": "text", "text": "Please describe this image in detail."},
            ],
        },
        {"role": "assistant", "content": description},
    ]

    chat_history.append({"role": "assistant", "content": f"**Auto-description:**\n\n{description}"})
    return chat_history, ""


def handle_chat(message: str, chat_history: list) -> tuple[str, list]:
    global api_history

    if not message.strip():
        return "", chat_history
    if not current_image_path:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "Please upload an image first."})
        return "", chat_history

    reply = ask_about_image(current_image_path, message, api_history)

    # Update API history for multi-turn context
    if not api_history:
        api_history.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _get_data_url(current_image_path)}},
                {"type": "text", "text": message},
            ],
        })
    else:
        api_history.append({"role": "user", "content": message})
    api_history.append({"role": "assistant", "content": reply})

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})
    return "", chat_history


def handle_clear() -> tuple[list, str]:
    global api_history
    api_history = []
    return [], ""


def _get_data_url(image_path: str) -> str:
    import base64, mimetypes
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


# --- UI ---

with gr.Blocks(title="NVIDIA NIM Vision Language Model") as demo:
    gr.Markdown("# NVIDIA NIM — Vision Language Model")
    gr.Markdown(
        "Upload an image and ask questions about it. "
        "Powered by **`meta/llama-3.2-11b-vision-instruct`** via NVIDIA NIM."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Image")
            image_input = gr.Image(type="filepath", label="Upload Image")
            status = gr.Textbox(label="Status", interactive=False, lines=1)
            auto_describe_btn = gr.Button("Auto-Describe Image", variant="secondary")

            gr.Markdown("**Try asking:**")
            gr.Markdown(
                "- What objects are in this image?\n"
                "- Is there any text visible?\n"
                "- What is the mood or setting?\n"
                "- Describe the colors used.\n"
                "- What is happening in this scene?"
            )

        with gr.Column(scale=2):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(height=450, label="Conversation")
            msg_input = gr.Textbox(
                placeholder="Ask something about the image...",
                label="Your question",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear Chat")

    image_input.change(fn=handle_image_upload, inputs=image_input, outputs=status)
    auto_describe_btn.click(fn=handle_auto_describe, inputs=chatbot, outputs=[chatbot, msg_input])
    submit_btn.click(fn=handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
    msg_input.submit(fn=handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
    clear_btn.click(fn=handle_clear, outputs=[chatbot, msg_input])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

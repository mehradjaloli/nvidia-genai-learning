"""
Vision Language Model (VLM) client
------------------------------------
Sends images + text to a NIM-hosted Vision Language Model.

Key difference from text-only LLMs:
  - Message content is a LIST of parts: one image part + one text part
  - Image is base64-encoded and sent inline as a data URL
  - Same OpenAI SDK, same base_url — just a different content structure

Model: meta/llama-3.2-11b-vision-instruct
  - Understands both images and text
  - Can describe scenes, read text in images, answer visual questions,
    compare objects, identify charts/diagrams, and more
"""

import os
import base64
import mimetypes
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

MODEL = "meta/llama-3.2-11b-vision-instruct"


def encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_string, mime_type)."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return b64, mime_type


def ask_about_image(image_path: str, question: str, history: list[dict]) -> str:
    """
    Send an image + question to the VLM and return the response.

    The image is attached to the FIRST user message only.
    Follow-up questions in the same conversation reference the same image
    via the conversation history — no need to re-send the image each turn.
    """
    b64, mime_type = encode_image(image_path)

    # First message includes the image; follow-ups are text only
    if not history:
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            },
            {
                "type": "text",
                "text": question,
            },
        ]
    else:
        user_content = question

    messages = history + [{"role": "user", "content": user_content}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )

    return response.choices[0].message.content


def describe_image(image_path: str) -> str:
    """Auto-generate a detailed description of the image."""
    return ask_about_image(
        image_path,
        question="Please describe this image in detail. Include objects, people, text, colors, and any other relevant information.",
        history=[],
    )

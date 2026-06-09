"""
Vision Client (Image understanding)
--------------------------------------
Handles single image Q&A, multi-image comparison, and image description.
Used by the orchestrator when the query involves images.
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

VLM_MODEL = "meta/llama-3.2-11b-vision-instruct"


def encode_image(image_path: str) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime_type


def image_part(image_path: str) -> dict:
    """Build an image_url content part from a file path."""
    b64, mime = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def ask_about_image(image_path: str, question: str, history: list[dict]) -> str:
    """Single image Q&A with multi-turn conversation support."""
    if not history:
        content = [image_part(image_path), {"type": "text", "text": question}]
    else:
        content = question

    messages = history + [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content


def compare_images(image_path_1: str, image_path_2: str, question: str) -> str:
    """
    Multi-image comparison.

    NOTE: Llama 3.2 Vision on NVIDIA NIM supports only 1 image per API call.
    Workaround: describe each image separately with two VLM calls, then
    pass both descriptions to the LLM for comparison reasoning.

    Pattern:
      Image 1 → VLM → Description 1 ──┐
                                        ├──► LLM → Comparison answer
      Image 2 → VLM → Description 2 ──┘
    """
    desc1 = ask_about_image(
        image_path_1,
        "Describe this image in full detail. Include objects, colors, layout, text, and any notable elements.",
        history=[],
    )
    desc2 = ask_about_image(
        image_path_2,
        "Describe this image in full detail. Include objects, colors, layout, text, and any notable elements.",
        history=[],
    )

    comparison_prompt = f"""You are comparing two images based on their descriptions below.

Image 1:
{desc1}

Image 2:
{desc2}

Question: {question}

Answer the question by comparing the two images based on their descriptions above."""

    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": comparison_prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    return (
        response.choices[0].message.content
        + "\n\n---\n**Image 1 description:**\n" + desc1
        + "\n\n**Image 2 description:**\n" + desc2
    )


def describe_image(image_path: str) -> str:
    return ask_about_image(
        image_path,
        question="Describe this image in detail. Include objects, people, text, colors, layout, and any notable visual elements.",
        history=[],
    )

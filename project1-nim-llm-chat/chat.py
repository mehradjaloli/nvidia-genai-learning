"""
Project 1: LLM Chat with NVIDIA NIM
------------------------------------
NIM (NVIDIA Inference Microservices) exposes optimized models as OpenAI-compatible
REST APIs. The only thing that changes between deployment targets is `base_url`.
Everything else — the SDK, model calls, message format — stays identical.

DEPLOYMENT OPTIONS (change base_url + api_key accordingly):

  1. NVIDIA Cloud (no GPU needed, free tier):
       base_url = "https://integrate.api.nvidia.com/v1"
       api_key  = your NGC API key from build.nvidia.com

  2. Local machine (requires NVIDIA GPU + Docker):
       First run the NIM container:
         docker run --gpus all -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
       Then:
         base_url = "http://localhost:8000/v1"
         api_key  = "not-required"  # any string, auth is disabled locally

  3. On-prem server (enterprise GPU server on your network):
       Same as local, but replace localhost with your server's IP:
         base_url = "http://YOUR_SERVER_IP:8000/v1"
         api_key  = "not-required"

The model name stays the same across all three. Your app code never changes.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- Switch deployment target by changing these two lines only ---
BASE_URL = "https://integrate.api.nvidia.com/v1"   # Option 1: NVIDIA Cloud
# BASE_URL = "http://localhost:8000/v1"             # Option 2: Local NIM container
# BASE_URL = "http://YOUR_SERVER_IP:8000/v1"        # Option 3: On-prem NIM server

API_KEY = os.environ["NVIDIA_API_KEY"]              # Option 1: NGC key from .env
# API_KEY = "not-required"                          # Option 2 & 3: local/on-prem

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

def chat(user_message: str, history: list[dict]) -> str:
    history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=history,
        temperature=0.7,
        max_tokens=512,
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


def main():
    print("NVIDIA NIM Chat — model: meta/llama-3.1-8b-instruct")
    print("Type 'quit' to exit\n")

    history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        reply = chat(user_input, history)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()

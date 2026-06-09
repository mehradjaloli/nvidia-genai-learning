# Project 1 — LLM Chat with NVIDIA NIM

## What we built
A simple CLI chat app that connects to a Llama 3.1 8B model hosted on NVIDIA's cloud,
using the NVIDIA NIM (Inference Microservices) API.

## What is NVIDIA NIM?
NIM is NVIDIA's service for running optimized AI models as REST APIs.
- Models run on NVIDIA GPUs in the cloud (NGC — NVIDIA GPU Cloud)
- API is OpenAI-compatible, so the same SDK works with just two line changes
- Under the hood, models are optimized with TensorRT-LLM for maximum throughput

## How it works
1. Created an account at [build.nvidia.com](https://build.nvidia.com) and generated an API key
2. Stored the key in a `.env` file at the project root (never commit this)
3. `chat.py` loads the key, connects to `integrate.api.nvidia.com`, and calls Llama 3.1 8B
4. Uses the OpenAI Python SDK — because NIM's API is OpenAI-compatible

## Key concept
The model (Llama 3.1 8B) is open-source from Meta, but it's **hosted and optimized by NVIDIA**
on their cloud infrastructure. You are not talking to ChatGPT or OpenAI at all.

## Deployment options — only `base_url` changes

| Target | `base_url` | `api_key` | Requires |
|---|---|---|---|
| **NVIDIA Cloud** (what we use) | `https://integrate.api.nvidia.com/v1` | NGC API key | Nothing local |
| **Local machine** | `http://localhost:8000/v1` | `"not-required"` | NVIDIA GPU + Docker |
| **On-prem server** | `http://YOUR_SERVER_IP:8000/v1` | `"not-required"` | NVIDIA GPU server on your network |

For local or on-prem, first start the NIM container on the GPU machine:
```bash
docker run --gpus all -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```
Then just change `BASE_URL` in `chat.py`. Everything else stays the same.

## Project structure
```
project1-nim-chat/
├── chat.py       # main chat script
└── README.md     # this file

../.env           # API key (one level up, never committed to git)
```

## Setup
```bash
pip install openai python-dotenv
```

Create a `.env` file one level above this folder:
```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
```

## Run
```bash
python3 chat.py
```

## Model used
`meta/llama-3.1-8b-instruct` — browse all available models at [build.nvidia.com/explore/reasoning](https://build.nvidia.com/explore/reasoning)

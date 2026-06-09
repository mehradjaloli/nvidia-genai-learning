# Project 3 — Vision Language Model with NVIDIA NIM

## What we built
A Gradio app for multi-turn visual conversations. Upload any image and ask
questions about it in natural language.

## What is a Vision Language Model (VLM)?
A VLM understands both images and text. It can:
- Describe scenes, objects, and people
- Read and extract text visible in images (OCR)
- Answer questions about charts, diagrams, and screenshots
- Compare and reason about visual content

## Model used
`meta/llama-3.2-11b-vision-instruct` — Llama 3.2's vision variant, hosted on NVIDIA NIM.

## Key concept: how images are sent to the API
Unlike text-only LLMs, VLMs receive a **list of content parts** in each message:

```python
{
    "role": "user",
    "content": [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,<base64_string>"}
        },
        {
            "type": "text",
            "text": "What is in this image?"
        }
    ]
}
```

The image is base64-encoded and sent inline as a data URL.
Follow-up questions are plain text — the model retains the image in context.

## Project structure
```
project3-nim-vision-language/
├── vision_client.py   # image encoding, API calls, multi-turn logic
├── app.py             # Gradio UI
└── README.md          # this file
```

## Setup
No additional packages needed beyond the base requirements:
```bash
pip install openai python-dotenv gradio
```

## Run
```bash
cd project3-nim-vision-language
python3 app.py
```

## Deployment options
Same as all NIM projects — change `base_url` only:

| Target | `base_url` |
|---|---|
| NVIDIA Cloud | `https://integrate.api.nvidia.com/v1` |
| Local NIM container | `http://localhost:8000/v1` |
| On-prem server | `http://YOUR_SERVER_IP:8000/v1` |

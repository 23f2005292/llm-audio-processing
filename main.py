import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):

    # Ask Gemini directly — it knows popular YouTube videos by URL
    prompt = f"""You are a YouTube video expert assistant.

A user wants to find when a specific topic is discussed in this YouTube video:
URL: {request.video_url}
Topic to find: "{request.topic}"

Based on your knowledge of this video, return the timestamp (HH:MM:SS) when this topic is FIRST spoken or discussed.

Return ONLY a valid JSON object like this:
{{"timestamp": "00:05:47"}}

Rules:
- Format must be HH:MM:SS (e.g. 00:05:47, 01:23:45)
- Return the FIRST occurrence
- If unsure, make your best estimate based on typical video structure
- Never return anything except the JSON object"""

    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": "google/gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
    )

    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    # Clean up in case model wraps in markdown
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    result = json.loads(content)
    timestamp = result["timestamp"]

    # Normalize to HH:MM:SS
    parts = timestamp.split(":")
    if len(parts) == 2:
        timestamp = f"00:{timestamp}"
    elif len(parts) == 1:
        secs = int(timestamp)
        timestamp = f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"

    return AskResponse(
        timestamp=timestamp,
        video_url=request.video_url,
        topic=request.topic
    )
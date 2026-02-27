import os
import time
import tempfile
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
import requests
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

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


def get_transcript(video_url: str) -> str:
    """Download subtitles/transcript from YouTube using yt-dlp (no ffmpeg needed)."""
    tmp_dir = tempfile.gettempdir()
    tmp_base = os.path.join(tmp_dir, f"yt_sub_{int(time.time())}")

    ydl_opts = {
        "skip_download": True,          # Don't download video or audio at all
        "writesubtitles": True,         # Download subtitles if available
        "writeautomaticsub": True,      # Use auto-generated subtitles if no manual ones
        "subtitleslangs": ["en"],       # English only
        "subtitlesformat": "json3",     # Machine-readable format
        "outtmpl": tmp_base,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Find the downloaded subtitle file
    for fname in os.listdir(tmp_dir):
        if fname.startswith(os.path.basename(tmp_base)) and fname.endswith(".json3"):
            sub_path = os.path.join(tmp_dir, fname)
            with open(sub_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            os.remove(sub_path)

            # Parse json3 format into "HH:MM:SS text" lines
            lines = []
            for event in data.get("events", []):
                if "segs" not in event:
                    continue
                start_ms = event.get("tStartMs", 0)
                text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
                if not text:
                    continue
                secs = int(start_ms // 1000)
                ts = f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
                lines.append(f"[{ts}] {text}")

            return "\n".join(lines)

    raise HTTPException(status_code=400, detail="No subtitles/transcript found for this video. Try a video with captions enabled.")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):

    # STEP A: Get transcript with timestamps
    transcript = get_transcript(request.video_url)

    # STEP B: Ask Gemini via AI Pipe (OpenRouter) to find the timestamp
    prompt = f"""Here is a transcript of a YouTube video with timestamps:

{transcript}

Find the FIRST timestamp where the topic "{request.topic}" is spoken or discussed.
Return ONLY a JSON object like: {{"timestamp": "00:05:47"}}
The timestamp must be in HH:MM:SS format. If not found, return {{"timestamp": "00:00:00"}}."""

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
    result = json.loads(content)
    timestamp = result["timestamp"]

    # STEP C: Normalize timestamp to HH:MM:SS
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
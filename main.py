import os
import time
import tempfile
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import yt_dlp

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
    """Download subtitles from YouTube with timestamps."""
    tmp_dir = tempfile.gettempdir()
    tmp_base = os.path.join(tmp_dir, f"yt_sub_{int(time.time())}")

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "json3",
        "outtmpl": tmp_base,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Find the subtitle file
    for fname in os.listdir(tmp_dir):
        if fname.startswith(os.path.basename(tmp_base)) and fname.endswith(".json3"):
            sub_path = os.path.join(tmp_dir, fname)
            with open(sub_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            os.remove(sub_path)

            # Parse into [HH:MM:SS] text lines
            lines = []
            for event in data.get("events", []):
                if "segs" not in event:
                    continue
                start_ms = event.get("tStartMs", 0)
                text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
                if not text or text == "\n":
                    continue
                secs = int(start_ms // 1000)
                ts = f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
                lines.append(f"[{ts}] {text}")

            return "\n".join(lines)

    raise HTTPException(status_code=400, detail="No subtitles found for this video.")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):

    # STEP 1: Get transcript with timestamps
    transcript = get_transcript(request.video_url)

    # STEP 2: Ask Gemini to find the exact timestamp
    prompt = f"""Here is a transcript of a YouTube video with timestamps in [HH:MM:SS] format:

{transcript[:60000]}

Find the FIRST timestamp where someone says or discusses: "{request.topic}"

Look for the closest matching text in the transcript and return its timestamp.
Return ONLY a JSON object: {{"timestamp": "HH:MM:SS"}}
Example: {{"timestamp": "01:14:37"}}
If not found, return {{"timestamp": "00:00:00"}}"""

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
    content = response.json()["choices"][0]["message"]["content"].strip()
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
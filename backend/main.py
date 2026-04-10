import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ai_agent import extract_youtube_id, find_viral_clip
from pipeline import run_pipeline

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TikTok Clip Factory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict[str, Any]] = {}


class GenerateRequest(BaseModel):
    mode: str = Field(pattern="^(tiktok|youtube)$")
    input: str = Field(min_length=1)


def _estimate_total_seconds(clip_duration: int) -> int:
    # Rough local pipeline estimate: download + crop/transcribe + subtitle burn/export.
    return max(90, int(75 + clip_duration * 2.2))


def _startup_checks() -> None:
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    try:
        import yt_dlp  # noqa: F401

        ytdlp_ok = True
    except Exception:
        ytdlp_ok = False
    if not ffmpeg_ok:
        print("[ERROR] ffmpeg was not found on PATH. Install: https://ffmpeg.org/download.html")
    if not ytdlp_ok:
        print("[ERROR] yt-dlp Python package missing. Install with: pip install yt-dlp")
    if not os.getenv("TENSORIX_API_KEY", "").strip():
        print("[INFO] TENSORIX_API_KEY is blank; running in fully local clip-selection mode.")


@app.on_event("startup")
async def on_startup() -> None:
    _startup_checks()


async def _run_job(job_id: str) -> None:
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["processing_started_at"] = time.time()
        await asyncio.to_thread(run_pipeline, jobs[job_id], OUTPUT_DIR)
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["completed_at"] = time.time()
    except Exception as exc:  # noqa: BLE001
        jobs[job_id]["status"] = "error"
        jobs[job_id]["completed_at"] = time.time()
        jobs[job_id]["error"] = str(exc)


@app.post("/api/generate")
async def generate_clip(body: GenerateRequest) -> JSONResponse:
    mode = body.mode
    input_text = body.input.strip()

    if mode == "youtube":
        try:
            _ = extract_youtube_id(input_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        clip_data = find_viral_clip(mode, input_text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"AI clip selection failed: {exc}") from exc

    job_id = str(uuid.uuid4())
    now = time.time()
    estimated_total_seconds = _estimate_total_seconds(int(clip_data["clip_duration"]))
    job = {
        "job_id": job_id,
        "mode": mode,
        "input": input_text,
        "status": "processing",
        "created_at": now,
        "processing_started_at": None,
        "completed_at": None,
        "estimated_total_seconds": estimated_total_seconds,
        "video_id": clip_data["video_id"],
        "title": clip_data["title"],
        "start_time": clip_data["start_time"],
        "clip_duration": min(60, int(clip_data["clip_duration"])),
        "viral_score": clip_data["viral_score"],
        "reason": clip_data["reason"],
        "output_path": None,
        "subtitle_words": [],
        "error": None,
    }
    jobs[job_id] = job

    asyncio.create_task(_run_job(job_id))

    return JSONResponse(
        {
            "job_id": job_id,
            "video_id": job["video_id"],
            "title": job["title"],
            "start_time": job["start_time"],
            "clip_duration": job["clip_duration"],
            "viral_score": job["viral_score"],
            "reason": job["reason"],
            "status": job["status"],
            "eta_seconds": estimated_total_seconds,
            "elapsed_seconds": 0,
        }
    )


@app.get("/api/status/{job_id}")
async def get_status(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    now = time.time()
    created_at = float(job.get("created_at") or now)
    elapsed_seconds = max(0, int(now - created_at))
    estimated_total_seconds = int(job.get("estimated_total_seconds") or 120)
    if job["status"] == "processing":
        eta_seconds = max(5, estimated_total_seconds - elapsed_seconds)
    else:
        eta_seconds = 0

    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "eta_seconds": eta_seconds,
            "elapsed_seconds": elapsed_seconds,
            "output_path": job.get("output_path"),
            "error": job.get("error"),
            "subtitle_words": job.get("subtitle_words", []),
            "video_id": job.get("video_id"),
            "start_time": job.get("start_time"),
            "title": job.get("title"),
            "viral_score": job.get("viral_score"),
            "reason": job.get("reason"),
        }
    )


@app.get("/api/download/{job_id}")
async def download_clip(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")

    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file missing")

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=Path(output_path).name,
    )

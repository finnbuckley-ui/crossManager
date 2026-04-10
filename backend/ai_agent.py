import re
from typing import Any, Dict, List

import yt_dlp

from pipeline import local_pick_clip_for_video

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})")


def extract_youtube_id(url_or_id: str) -> str:
    text = (url_or_id or "").strip()
    if len(text) == 11 and re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    match = YOUTUBE_ID_RE.search(text)
    if not match:
        raise ValueError("Could not extract a valid 11-character YouTube video ID from input.")
    return match.group(1)


def _search_candidates(niche_query: str, limit: int = 10) -> List[Dict[str, Any]]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extract_flat": "in_playlist",
        "default_search": "ytsearch",
    }
    query = f"ytsearch{limit}:{niche_query}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
    entries = info.get("entries") or []
    valid: List[Dict[str, Any]] = []
    for e in entries:
        video_id = str(e.get("id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            continue
        duration = int(e.get("duration") or 0)
        valid.append(
            {
                "video_id": video_id,
                "title": str(e.get("title") or "Untitled video"),
                "duration": duration,
                "view_count": int(e.get("view_count") or 0),
                "like_count": int(e.get("like_count") or 0),
                "uploader": str(e.get("uploader") or ""),
            }
        )
    return valid


def _metadata_priority(c: Dict[str, Any]) -> float:
    # Weighted metadata ranking before deep analysis.
    views = max(0, int(c.get("view_count", 0)))
    likes = max(0, int(c.get("like_count", 0)))
    duration = max(1, int(c.get("duration", 1)))

    engagement = (likes / max(views, 1)) if views else 0.0
    duration_fit = 1.0 - min(1.0, abs(duration - 480) / 900.0)
    popularity = min(1.0, (views / 2_000_000.0) ** 0.5)
    return 0.5 * popularity + 0.35 * min(1.0, engagement * 10.0) + 0.15 * duration_fit


def find_viral_clip(mode: str, input_text: str) -> Dict[str, Any]:
    if mode == "tiktok":
        candidates = _search_candidates(input_text, limit=10)
        if not candidates:
            raise ValueError("No suitable YouTube videos found for this niche.")

        ranked = sorted(candidates, key=_metadata_priority, reverse=True)[:3]

        best: Dict[str, Any] | None = None
        best_score = -1.0
        for candidate in ranked:
            clip = local_pick_clip_for_video(candidate["video_id"], preferred_title=candidate["title"])
            score = float(clip.get("viral_score", 0))
            if score > best_score:
                best = clip
                best_score = score

        if not best:
            raise ValueError("Could not determine a clip from local analysis.")
        return best
    elif mode == "youtube":
        video_id = extract_youtube_id(input_text)
        return local_pick_clip_for_video(video_id)
    else:
        raise ValueError("mode must be either 'tiktok' or 'youtube'")

import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import ffmpeg
import numpy as np
from PIL import ImageFont
import whisper
import yt_dlp

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

PUNCT_ONLY_RE = re.compile(r"^[^\w]+$")


def _escape_text_for_ffmpeg(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
    )


def _check_binary(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary '{name}' not found on PATH. {install_hint}")


def _next_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("videooutputnumber#*.mp4"))
    nums: List[int] = []
    for p in existing:
        stem = p.stem
        if "#" in stem:
            raw = stem.split("#")[-1]
            if raw.isdigit():
                nums.append(int(raw))
    next_num = (max(nums) + 1) if nums else 1
    return output_dir / f"videooutputnumber#{next_num}.mp4"


def _base_ytdlp_opts() -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Prefer clients that are often less dependent on JS challenge execution.
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    return opts


def _download_video(video_id: str, temp_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        **_base_ytdlp_opts(),
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]",
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = Path(temp_dir / f"{info['id']}.mp4")
        if downloaded_path.exists():
            return downloaded_path, info
        requested = info.get("requested_downloads") or []
        if requested and requested[0].get("filepath"):
            return Path(requested[0]["filepath"]), info

    raise RuntimeError("Failed to download video with yt-dlp.")


def _extract_audio_wav(video_path: Path, wav_path: Path) -> None:
    (
        ffmpeg.input(str(video_path))
        .output(str(wav_path), ac=1, ar=16000, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )


def _audio_energy_series(wav_path: Path, hop_seconds: float = 0.5) -> Tuple[np.ndarray, float]:
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, int(sample_rate * hop_seconds))
    if len(samples) < hop:
        return np.array([0.0], dtype=np.float32), hop_seconds

    energies: List[float] = []
    for i in range(0, len(samples) - hop + 1, hop):
        frame = samples[i : i + hop]
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        energies.append(rms)

    arr = np.array(energies, dtype=np.float32)
    if arr.size and float(arr.max()) > 0:
        arr = arr / float(arr.max())
    return arr, hop_seconds


def _video_dynamics_series(video_path: Path, sample_fps: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for local clip analysis.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    times: List[float] = []
    motion: List[float] = []
    shot_change: List[float] = []
    face_presence: List[float] = []
    face_area: List[float] = []
    face_center_offset: List[float] = []

    prev_gray = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            t = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
            face_count = len(faces)
            face_presence.append(float(min(1.0, face_count / 2.0)))

            if face_count > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_area.append(float((w * h) / max(1, frame.shape[0] * frame.shape[1])))
                face_cx = float(x + (w / 2.0))
                face_center_offset.append(abs(face_cx - (frame.shape[1] / 2.0)) / max(1.0, frame.shape[1] / 2.0))
            else:
                face_area.append(0.0)
                face_center_offset.append(1.0)

            if prev_gray is None:
                diff_mean = 0.0
            else:
                diff = cv2.absdiff(prev_gray, gray)
                diff_mean = float(np.mean(diff) / 255.0)
            motion.append(diff_mean)
            shot_change.append(1.0 if diff_mean > 0.18 else 0.0)
            times.append(t)

            prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()

    return (
        np.array(times, dtype=np.float32),
        np.array(motion, dtype=np.float32),
        np.array(shot_change, dtype=np.float32),
        np.array(face_presence, dtype=np.float32),
        np.array(face_area, dtype=np.float32),
        np.array(face_center_offset, dtype=np.float32),
    )


def _window_mean(series: np.ndarray, idx0: int, idx1: int) -> float:
    if series.size == 0 or idx1 <= idx0:
        return 0.0
    return float(np.mean(series[idx0:idx1]))


def _pick_best_window(
    duration_s: float,
    times: np.ndarray,
    motion: np.ndarray,
    shots: np.ndarray,
    faces: np.ndarray,
    face_area: np.ndarray,
    face_center_offset: np.ndarray,
    audio: np.ndarray,
    audio_hop_s: float,
) -> Tuple[int, int, int, str]:
    if duration_s <= 35:
        return 0, int(max(30, min(35, duration_s))), 6, "Short source video; selected the strongest available continuous segment."

    clip_duration = int(max(30, min(60, 50 if duration_s >= 70 else duration_s * 0.7)))
    max_start = max(0, int(duration_s) - clip_duration)
    if max_start <= 0:
        return 0, clip_duration, 6, "Selected the highest-intensity section from a short source video."

    best_start = 0
    best_score = -1.0

    for start in range(0, max_start + 1, 3):
        end = start + clip_duration

        video_mask = (times >= start) & (times < end)
        if not np.any(video_mask):
            continue
        vm = motion[video_mask]
        vs = shots[video_mask]
        vf = faces[video_mask]
        vaa = face_area[video_mask]
        vco = face_center_offset[video_mask]

        a0 = int(start / audio_hop_s)
        a1 = max(a0 + 1, int(end / audio_hop_s))
        a1 = min(a1, audio.size)
        va = audio[a0:a1] if audio.size else np.array([0.0], dtype=np.float32)

        motion_mean = float(np.mean(vm))
        motion_std = float(np.std(vm))
        shot_density = float(np.mean(vs))
        face_mean = float(np.mean(vf))
        face_area_mean = float(np.mean(vaa))
        face_center_score = 1.0 - float(np.mean(vco))
        audio_mean = float(np.mean(va)) if va.size else 0.0
        audio_peak = float(np.percentile(va, 90)) if va.size else 0.0

        # Weighted for hooks + stable framing while penalizing chaotic footage.
        score = (
            0.35 * audio_mean
            + 0.17 * audio_peak
            + 0.18 * face_mean
            + 0.12 * face_area_mean
            + 0.10 * face_center_score
            + 0.10 * shot_density
            + 0.08 * motion_mean
            - 0.18 * motion_std
        )

        if score > best_score:
            best_score = score
            best_start = start

    viral_score = int(max(1, min(10, round(5 + best_score * 6))))
    reason = (
        "Locally selected for strong speech/audio energy and a stable, centered on-screen subject, with just enough visual change to keep retention high."
    )
    return best_start, clip_duration, viral_score, reason


def local_pick_clip_for_video(video_id: str, preferred_title: str | None = None) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tt_pick_") as tmp:
        temp_dir = Path(tmp)
        video_path, info = _download_video(video_id, temp_dir)

        title = preferred_title or str(info.get("title") or video_id)
        duration = float(info.get("duration") or 0.0)

        analysis_video = video_path
        if duration <= 0:
            probe = ffmpeg.probe(str(video_path))
            fmt = probe.get("format", {})
            duration = float(fmt.get("duration") or 0.0)

        wav_path = temp_dir / "analysis.wav"
        _extract_audio_wav(analysis_video, wav_path)
        audio_series, audio_hop_s = _audio_energy_series(wav_path, hop_seconds=0.5)

        times, motion, shots, faces, face_area, face_center_offset = _video_dynamics_series(analysis_video, sample_fps=2.0)
        if duration <= 0 and times.size:
            duration = float(times[-1])

        start_time, clip_duration, viral_score, reason = _pick_best_window(
            duration_s=max(1.0, duration),
            times=times,
            motion=motion,
            shots=shots,
            faces=faces,
            face_area=face_area,
            face_center_offset=face_center_offset,
            audio=audio_series,
            audio_hop_s=audio_hop_s,
        )

    return {
        "video_id": video_id,
        "title": title,
        "start_time": int(start_time),
        "clip_duration": int(max(30, min(60, clip_duration))),
        "viral_score": int(max(1, min(10, viral_score))),
        "reason": reason,
    }


def _trim_clip(input_path: Path, output_path: Path, start_time: int, duration: int) -> None:
    (
        ffmpeg.input(str(input_path), ss=start_time, t=duration)
        .output(str(output_path), c="copy")
        .overwrite_output()
        .run(quiet=True)
    )


def _detect_subject_x(frame: np.ndarray, prev_gray: np.ndarray | None) -> Tuple[int, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

    if len(faces) > 0:
        x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])
        return int(x + w / 2), gray

    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (9, 9), 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, _, w, _ = cv2.boundingRect(c)
            return int(x + w / 2), gray

    return int(frame.shape[1] / 2), gray


def _rolling_average(values: List[float], window: int) -> List[float]:
    out: List[float] = []
    dq: deque[float] = deque(maxlen=window)
    for v in values:
        dq.append(v)
        out.append(float(sum(dq)) / len(dq))
    return out


def _compute_crop_positions(
    clip_path: Path,
) -> Tuple[List[int], int, int, int, int, float, int]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open trimmed clip for crop analysis.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_every = max(1, int(round(fps / 2.0)))  # 2 fps sampling
    sample_indices: List[int] = []
    sample_x: List[float] = []

    prev_gray = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            subject_x, prev_gray = _detect_subject_x(frame, prev_gray)
            sample_indices.append(frame_idx)
            sample_x.append(float(subject_x))
        else:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

    cap.release()

    if not sample_indices:
        sample_indices = [0, max(0, total_frames - 1)]
        sample_x = [width / 2.0, width / 2.0]

    smooth_x = _rolling_average(sample_x, window=10)
    std_x = float(np.std(smooth_x)) if smooth_x else 0.0

    target_crop_w = int(min(width, height * 9 / 16))
    target_crop_h = int(round(target_crop_w * 16 / 9))
    if target_crop_h > height:
        target_crop_h = height
        target_crop_w = int(round(target_crop_h * 9 / 16))

    if std_x < 50.0:
        median_x = float(np.median(smooth_x))
        x_left = int(max(0, min(width - target_crop_w, median_x - target_crop_w / 2)))
        return [x_left] * max(1, total_frames), target_crop_w, target_crop_h, width, height, fps, total_frames

    frame_numbers = np.arange(max(1, total_frames), dtype=np.float32)
    interp_x_center = np.interp(frame_numbers, np.array(sample_indices, dtype=np.float32), np.array(smooth_x, dtype=np.float32))

    x_positions: List[int] = []
    for center_x in interp_x_center:
        left = int(round(center_x - target_crop_w / 2))
        left = max(0, min(width - target_crop_w, left))
        x_positions.append(left)

    return x_positions, target_crop_w, target_crop_h, width, height, fps, total_frames


def _apply_dynamic_crop(
    trimmed_path: Path,
    cropped_video_path: Path,
    x_positions: List[int],
    crop_w: int,
    crop_h: int,
    source_h: int,
    fps: float,
) -> None:
    cap = cv2.VideoCapture(str(trimmed_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open trimmed clip for cropping.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cropped_video_path), fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to initialize video writer for cropped clip.")

    y_top = max(0, int((source_h - crop_h) / 2))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x = x_positions[min(frame_idx, len(x_positions) - 1)]
        x = max(0, min(frame.shape[1] - crop_w, x))
        crop = frame[y_top : y_top + crop_h, x : x + crop_w]
        if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            crop = cv2.resize(crop, (crop_w, crop_h))
        writer.write(crop)
        frame_idx += 1

    cap.release()
    writer.release()


def _mux_audio(video_path: Path, audio_source_path: Path, output_path: Path) -> None:
    v = ffmpeg.input(str(video_path)).video
    a = ffmpeg.input(str(audio_source_path)).audio
    (
        ffmpeg.output(v, a, str(output_path), vcodec="libx264", acodec="aac", audio_bitrate="192k", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )


def _normalize_to_portrait_canvas(input_path: Path, output_path: Path) -> None:
    # Lock final composition space to portrait so subtitle coordinates are always stable.
    vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    (
        ffmpeg.input(str(input_path))
        .output(
            str(output_path),
            vf=vf,
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="192k",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run(quiet=True)
    )


def _flatten_word_timestamps(transcribe_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_words: List[Dict[str, Any]] = []
    for segment in transcribe_result.get("segments", []):
        for word in segment.get("words", []) or []:
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            raw_words.append(
                {
                    "word": text,
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                }
            )
    raw_words.sort(key=lambda w: w["start"])

    merged: List[Dict[str, Any]] = []
    pending_prefix = ""
    for token in raw_words:
        text = token["word"]
        if PUNCT_ONLY_RE.fullmatch(text):
            if merged:
                merged[-1]["word"] += text
                merged[-1]["end"] = max(float(merged[-1]["end"]), float(token["end"]))
            else:
                pending_prefix += text
            continue

        if pending_prefix:
            text = pending_prefix + text
            pending_prefix = ""

        merged.append(
            {
                "word": text,
                "start": float(token["start"]),
                "end": float(token["end"]),
            }
        )

    if pending_prefix and merged:
        merged[-1]["word"] += pending_prefix

    return merged


def _estimate_text_px(text: str, font_size: int) -> int:
    # Kept as a fallback measurement path when exact font measurement is unavailable.
    return int(max(1, len(text)) * font_size * 0.58)


@lru_cache(maxsize=32)
def _load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, font_size)


def _measure_text_px(text: str, font_path: Path, font_size: int) -> int:
    font = _load_font(str(font_path), font_size)
    bbox = font.getbbox(text or " ")
    return int(bbox[2] - bbox[0])


def _rgb_to_ass_color(hex_color: str) -> str:
    clean = hex_color.lstrip("#")
    if len(clean) != 6:
        raise ValueError(f"Invalid color: {hex_color}")
    rr = clean[0:2]
    gg = clean[2:4]
    bb = clean[4:6]
    return f"&H{bb}{gg}{rr}&"


def _ass_time(seconds: float) -> str:
    total_centis = max(0, int(round(seconds * 100)))
    centis = total_centis % 100
    total_seconds = total_centis // 100
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh}:{mm:02d}:{ss:02d}.{centis:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _split_into_subtitle_lines(
    words: List[Dict[str, Any]],
    font_path: Path,
    font_size: int,
    max_width_px: int,
    max_words_per_line: int = 5,
    gap_threshold: float = 0.75,
) -> List[List[Dict[str, Any]]]:
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for idx, word in enumerate(words):
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            if gap >= gap_threshold:
                lines.append(current)
                current = []

        tentative = current + [word]
        tentative_text = " ".join(item["word"] for item in tentative)
        tentative_width = _measure_text_px(tentative_text, font_path, font_size)

        should_break = False
        if current:
            should_break = (
                len(tentative) > max_words_per_line
                or tentative_width > max_width_px
            )

        if should_break:
            lines.append(current)
            current = [word]
        else:
            current = tentative

        if current and current[-1]["word"].endswith((".", "?", "!")):
            next_gap = None
            if idx + 1 < len(words):
                next_gap = float(words[idx + 1]["start"]) - float(current[-1]["end"])
            if next_gap is None or next_gap >= 0:
                lines.append(current)
                current = []

    if current:
        lines.append(current)

    return lines


def _fit_line_font_size(words: List[Dict[str, Any]], font_path: Path, preferred_size: int, max_width_px: int) -> int:
    size = preferred_size
    line_text = " ".join(item["word"] for item in words)
    while size > 24 and _measure_text_px(line_text, font_path, size) > max_width_px:
        size -= 1
    return size


def _build_ass_subtitles(words: List[Dict[str, Any]], width: int, height: int) -> str:
    if not words:
        return ""

    font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    if not font_path.exists():
        font_path = Path("C:/Windows/Fonts/arial.ttf")

    base_font_size = max(42, int(height * 0.034))
    max_text_width = int(width * 0.84)
    lines = _split_into_subtitle_lines(words, font_path, base_font_size, max_text_width)

    outline_color = _rgb_to_ass_color("#000000")
    grey_color = _rgb_to_ass_color("#AAAAAA")
    yellow_color = _rgb_to_ass_color("#FFE500")
    white_color = _rgb_to_ass_color("#FFFFFF")
    box_color = _rgb_to_ass_color("#000000")

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Default,Arial,{base_font_size},{grey_color},{yellow_color},{outline_color},{box_color},1,0,0,0,100,100,0,0,1,3,0,8,80,80,120,1",
        f"Style: Box,Arial,{base_font_size},{box_color},{box_color},{outline_color},{box_color},1,0,0,0,100,100,0,0,3,0,0,8,80,80,120,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    events: List[str] = []
    base_y = int(height * 0.74)

    for line_index, line_words in enumerate(lines):
        font_size = _fit_line_font_size(line_words, font_path, base_font_size, max_text_width)
        line_text = " ".join(item["word"] for item in line_words)
        center_x = int(width / 2)
        box_y = base_y
        text_y = box_y

        line_start = float(line_words[0]["start"])
        natural_end = float(line_words[-1]["end"] + 0.06)
        if line_index + 1 < len(lines):
            next_start = float(lines[line_index + 1][0]["start"])
            line_end = min(natural_end, next_start - 0.04)
        else:
            line_end = natural_end
        line_end = max(line_start + 0.04, line_end)
        line_start_text = _ass_time(line_start)
        line_end_text = _ass_time(line_end)

        events.append(
            f"Dialogue: 0,{line_start_text},{line_end_text},Box,,0,0,0,,"
            f"{{\\an8\\pos({center_x},{box_y})\\fs{font_size}\\bord0\\shad0\\1c{box_color}\\1a&H66&}}{_ass_escape(line_text)}"
        )

        line_total_cs = max(1, int(round((line_end - line_start) * 100)))
        word_durations = [max(4, int(round((float(item["end"]) - float(item["start"])) * 100))) for item in line_words]
        duration_sum = sum(word_durations) or len(word_durations)
        scale = line_total_cs / duration_sum
        scaled = [max(4, int(round(value * scale))) for value in word_durations]
        correction = line_total_cs - sum(scaled)
        if scaled:
            scaled[-1] += correction

        text_parts: List[str] = []
        for word_index, (item, k_cs) in enumerate(zip(line_words, scaled)):
            if word_index > 0:
                text_parts.append(" ")
            text_parts.append(f"{{\\k{k_cs}}}{_ass_escape(item['word'])}")

        events.append(
            f"Dialogue: 1,{line_start_text},{line_end_text},Default,,0,0,0,,"
            f"{{\\an8\\pos({center_x},{text_y})\\fs{font_size}\\bord3\\shad0\\b1\\c{grey_color}\\2c{yellow_color}\\3c{outline_color}\\4c{white_color}}}"
            + "".join(text_parts)
        )

    return "\n".join(header + events) + "\n"


def _build_lines(words: List[Dict[str, Any]], max_words_per_line: int, max_width_px: int, font_size: int) -> List[List[Dict[str, Any]]]:
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for word in words:
        tentative = current + [word]
        text = " ".join(w["word"] for w in tentative)
        if (
            current
            and (
                len(tentative) > max_words_per_line
                or _estimate_text_px(text, font_size) > max_width_px
            )
        ):
            lines.append(current)
            current = [word]
        else:
            current = tentative
    if current:
        lines.append(current)
    return lines


def _fit_font_size(line_words: List[Dict[str, Any]], width: int, preferred_size: int, font_path: Path) -> int:
    size = preferred_size
    max_line_width = int(width * 0.84)
    text = " ".join(w["word"] for w in line_words)
    while size > 22 and _measure_text_px(text, font_path, size) > max_line_width:
        size -= 1
    return size


def _burn_karaoke_subtitles(cropped_av_path: Path, words: List[Dict[str, Any]], output_path: Path) -> None:
    probe = ffmpeg.probe(str(cropped_av_path))
    video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        raise RuntimeError("No video stream found in cropped clip.")

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    ass_text = _build_ass_subtitles(words, width, height)
    if not ass_text:
        raise RuntimeError("No subtitle text could be generated.")

    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False, encoding="utf-8", newline="\n") as script_file:
            script_file.write(ass_text)
            script_path = Path(script_file.name)

        safe_ass_path = str(script_path).replace("\\", "/").replace(":", "\\:")
        vf = f"subtitles='{safe_ass_path}'"

        (
            ffmpeg.input(str(cropped_av_path))
            .output(
                str(output_path),
                vf=vf,
                vcodec="libx264",
                crf=23,
                acodec="aac",
                audio_bitrate="192k",
                pix_fmt="yuv420p",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    finally:
        if script_path and script_path.exists():
            script_path.unlink(missing_ok=True)


def run_pipeline(job: Dict[str, Any], output_dir: Path) -> None:
    _check_binary("ffmpeg", "Install from https://ffmpeg.org/download.html")
    try:
        import yt_dlp  # noqa: F401
    except Exception as exc:
        raise RuntimeError("yt-dlp Python package is required. Install with pip install yt-dlp") from exc

    video_id = str(job["video_id"])
    start_time = int(job["start_time"])
    clip_duration = int(job["clip_duration"])

    with tempfile.TemporaryDirectory(prefix="tt_factory_") as tmp:
        temp_dir = Path(tmp)
        full_video, _ = _download_video(video_id, temp_dir)

        trimmed = temp_dir / "trimmed.mp4"
        _trim_clip(full_video, trimmed, start_time, clip_duration)

        x_positions, crop_w, crop_h, _, source_h, fps, _ = _compute_crop_positions(trimmed)

        cropped_video_no_audio = temp_dir / "cropped_no_audio.mp4"
        _apply_dynamic_crop(trimmed, cropped_video_no_audio, x_positions, crop_w, crop_h, source_h, fps)

        cropped_with_audio = temp_dir / "cropped_with_audio.mp4"
        _mux_audio(cropped_video_no_audio, trimmed, cropped_with_audio)

        portrait_ready = temp_dir / "portrait_ready.mp4"
        _normalize_to_portrait_canvas(cropped_with_audio, portrait_ready)

        whisper_model = whisper.load_model("base")
        transcribe_result = whisper_model.transcribe(str(portrait_ready), word_timestamps=True)
        words = _flatten_word_timestamps(transcribe_result)

        final_output = _next_output_path(output_dir)
        _burn_karaoke_subtitles(portrait_ready, words, final_output)

        job["subtitle_words"] = words
        job["output_path"] = str(final_output)

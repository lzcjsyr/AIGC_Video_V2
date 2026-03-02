"""YouTube search/download/clip helpers based on yt-dlp and ffmpeg."""

from __future__ import annotations

import glob
import os
import subprocess
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

try:
    import yt_dlp
except Exception:  # pragma: no cover - fallback handled at runtime
    yt_dlp = None


def is_valid_youtube_url(url: str) -> bool:
    value = (url or "").strip()
    if not value:
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def extract_video_id(url: str) -> str:
    if not is_valid_youtube_url(url):
        return ""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        return parsed.path.strip("/")

    if parsed.path == "/watch":
        query = parse_qs(parsed.query)
        return (query.get("v") or [""])[0]

    # 兼容 /shorts/<id> 等路径
    parts = [part for part in parsed.path.split("/") if part]
    return parts[-1] if parts else ""


def _ensure_yt_dlp() -> None:
    if yt_dlp is None:
        raise RuntimeError("缺少 yt-dlp 依赖，请安装 requirements.txt")


def search_youtube_candidates(
    query: str,
    limit: int = 8,
    min_duration_seconds: Optional[int] = None,
    max_duration_seconds: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search public YouTube videos and return normalized candidates."""
    _ensure_yt_dlp()

    q = (query or "").strip()
    if not q:
        return []

    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "noplaylist": True,
    }

    search_expr = f"ytsearch{max(1, int(limit))}:{q}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        data = ydl.extract_info(search_expr, download=False)

    entries = (data or {}).get("entries") or []
    candidates: List[Dict[str, Any]] = []

    for item in entries:
        duration = item.get("duration")
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except ValueError:
                duration = None

        if min_duration_seconds is not None and isinstance(duration, (int, float)):
            if float(duration) < float(min_duration_seconds):
                continue
        if max_duration_seconds is not None and isinstance(duration, (int, float)):
            if float(duration) > float(max_duration_seconds):
                continue

        video_id = str(item.get("id") or "").strip()
        webpage_url = str(item.get("webpage_url") or "").strip()
        if not webpage_url and video_id:
            webpage_url = f"https://www.youtube.com/watch?v={video_id}"
        if not is_valid_youtube_url(webpage_url):
            continue

        candidates.append(
            {
                "video_id": video_id,
                "title": str(item.get("title") or "").strip(),
                "duration": float(duration) if isinstance(duration, (int, float)) else 0.0,
                "youtube_url": webpage_url,
                "channel": str(item.get("channel") or item.get("uploader") or "").strip(),
            }
        )

    return candidates


def _locate_downloaded_file(cache_dir: str, video_id: str, fallback_path: str) -> str:
    if os.path.exists(fallback_path):
        return fallback_path

    if video_id:
        patterns = [
            os.path.join(cache_dir, f"{video_id}.mp4"),
            os.path.join(cache_dir, f"{video_id}.mkv"),
            os.path.join(cache_dir, f"{video_id}.webm"),
            os.path.join(cache_dir, f"{video_id}.*"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                # 优先选择mp4
                matches.sort(key=lambda p: (0 if p.lower().endswith(".mp4") else 1, p))
                return matches[0]

    raise FileNotFoundError("未找到下载后的视频文件")


def download_youtube_video(youtube_url: str, cache_dir: str) -> Dict[str, Any]:
    """Download a YouTube video into cache dir and return metadata."""
    _ensure_yt_dlp()

    if not is_valid_youtube_url(youtube_url):
        raise ValueError(f"无效YouTube链接: {youtube_url}")

    os.makedirs(cache_dir, exist_ok=True)

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(cache_dir, "%(id)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        fallback_path = ydl.prepare_filename(info)

    video_id = str((info or {}).get("id") or extract_video_id(youtube_url) or "").strip()
    local_path = _locate_downloaded_file(cache_dir, video_id, fallback_path)

    return {
        "video_id": video_id,
        "title": str((info or {}).get("title") or "").strip(),
        "duration": float((info or {}).get("duration") or 0.0),
        "local_path": local_path,
    }


def probe_video_duration(video_path: str) -> float:
    """Probe video duration via ffprobe."""
    if not os.path.exists(video_path):
        return 0.0

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float((result.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def trim_video_clip(input_path: str, output_path: str, start_seconds: float, end_seconds: float) -> str:
    """Trim and re-encode video clip without audio."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"视频文件不存在: {input_path}")

    start = max(0.0, float(start_seconds or 0.0))
    end = max(0.0, float(end_seconds or 0.0))
    if end <= start:
        raise ValueError(f"非法裁剪区间: start={start}, end={end}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        output_path,
    ]

    subprocess.run(cmd, check=True)
    return output_path

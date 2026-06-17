"""
Phase 1: Download audio from YouTube sources listed in config/sources.csv.
Outputs downloaded files to data/raw/ and logs metadata to metadata/source_log.csv.

Usage:
    python scripts/01_download.py              # Download all sources
    python scripts/01_download.py --index 0    # Download only source at index 0
    python scripts/01_download.py --resume     # Skip already-downloaded sources
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
SOURCES_CSV = ROOT / "config" / "sources.csv"
SOURCE_LOG_CSV = ROOT / "metadata" / "source_log.csv"
RAW_DIR = ROOT / "data" / "raw"

SOURCE_LOG_FIELDS = [
    "source_index", "language", "genre", "speaker_name",
    "channel_name", "video_url", "video_id", "title",
    "duration_seconds", "download_timestamp", "local_file",
    "target_speaker_label",
]


def get_video_id(url: str) -> str:
    patterns = [
        r"youtu\.be/([^?&/\s]+)",
        r"youtube\.com/watch\?v=([^&\s]+)",
        r"youtube\.com/v/([^?&\s]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return ""


def read_sources() -> list[dict]:
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_source_log() -> list[dict]:
    if not SOURCE_LOG_CSV.exists():
        return []
    with open(SOURCE_LOG_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_source_log(rows: list[dict]) -> None:
    SOURCE_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCE_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def download_source(row: dict, index: int) -> dict | None:
    url = row["video_url"].strip()
    if not url or url.upper() == "TBD":
        print(f"  [SKIP] Row {index}: URL is TBD or empty")
        return None

    lang = row["language"].strip()
    genre = row["genre"].strip()
    output_stem = f"{lang}_{genre}_{index:03d}"
    output_template = str(RAW_DIR / f"{output_stem}.%(ext)s")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": False,
        "postprocessors": [],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        ext = info.get("ext", "webm")
        local_file = RAW_DIR / f"{output_stem}.{ext}"

        # Fall back to searching if the expected path doesn't match
        if not local_file.exists():
            candidates = sorted(RAW_DIR.glob(f"{output_stem}.*"))
            local_file = candidates[0] if candidates else None

        duration = info.get("duration", 0) or 0
        title = info.get("title", "Unknown")
        channel = info.get("channel") or info.get("uploader", "Unknown")

        print(f"  [OK] '{title}' | {duration}s | channel: {channel}")

        return {
            "source_index": index,
            "language": lang,
            "genre": genre,
            "speaker_name": row.get("speaker_name", ""),
            "channel_name": row.get("channel_name", ""),
            "video_url": url,
            "video_id": get_video_id(url) or info.get("id", ""),
            "title": title,
            "duration_seconds": duration,
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "local_file": str(local_file) if local_file else "",
            "target_speaker_label": "",
        }

    except yt_dlp.utils.DownloadError as e:
        print(f"  [FAIL] {url}: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] {url}: {type(e).__name__}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download audio from YouTube sources")
    parser.add_argument("--index", type=int, default=None,
                        help="Process only the source at this 0-based index")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sources already present in source_log.csv")
    args = parser.parse_args()

    sources = read_sources()
    log_rows = load_source_log()
    existing_indices = {
        int(r["source_index"]) for r in log_rows
        if r.get("source_index", "").lstrip("-").isdigit() and r.get("local_file")
    }

    for i, row in enumerate(sources):
        if args.index is not None and i != args.index:
            continue
        if args.resume and i in existing_indices:
            print(f"[{i:02d}] Already downloaded, skipping: {row['video_url']}")
            continue

        speaker = row.get("speaker_name", "Unknown")
        print(f"[{i:02d}] {speaker} — {row['video_url']}")

        entry = download_source(row, i)
        if entry is not None:
            match_idx = next(
                (j for j, r in enumerate(log_rows) if str(r.get("source_index")) == str(i)),
                None,
            )
            if match_idx is not None:
                log_rows[match_idx] = entry
            else:
                log_rows.append(entry)
            save_source_log(log_rows)

    print(f"\nSource log written to: {SOURCE_LOG_CSV}")
    downloaded = sum(1 for r in log_rows if r.get("local_file"))
    print(f"Total downloaded: {downloaded}/{len(sources)}")


if __name__ == "__main__":
    main()

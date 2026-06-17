"""
Phase 4: Transcribe each clip using Sarvam's Batch STT API in verbatim mode.
Uses the async Batch API because clips are 45-60s (REST API limit is < 30s).
Updates the 'transcript' column in clips_metadata.csv after each clip.

IMPORTANT: Run --dry-run first to see cost estimate before processing the full batch.

Usage:
    python scripts/04_transcribe.py --dry-run                     # Cost estimate, first clip only
    python scripts/04_transcribe.py --clip en_news_000_0001.wav   # One clip
    python scripts/04_transcribe.py --all                         # All without transcripts
    python scripts/04_transcribe.py --all --redo                  # Re-transcribe everything
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
CLIPS_META_CSV = ROOT / "metadata" / "clips_metadata.csv"
CLIPS_DIR = ROOT / "data" / "clips"
SETTINGS = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())

SARVAM_BASE_URL = SETTINGS["sarvam"]["base_url"]
BATCH_MODEL = SETTINGS["sarvam"]["batch_model"]
SUBMIT_PATH = SETTINGS["sarvam"]["batch_submit_path"]
STATUS_PATH = SETTINGS["sarvam"]["batch_status_path"]
COST_PER_HOUR_INR = SETTINGS["sarvam"]["cost_per_hour_inr"]
SUBMIT_TIMEOUT = SETTINGS["sarvam"]["batch_submit_timeout_s"]
POLL_INTERVAL = SETTINGS["sarvam"]["batch_poll_interval_s"]
MAX_WAIT = SETTINGS["sarvam"]["batch_max_wait_s"]
LANG_CODES = SETTINGS["language_codes"]

CLIPS_META_FIELDS = [
    "clip_filename", "language", "genre", "speaker_name",
    "source_index", "video_id", "source_channel", "source_url",
    "clip_start_time_s", "clip_end_time_s", "duration_s",
    "transcript", "primary_emotion", "secondary_emotion",
    "approved", "reviewed", "qc_flags",
]


def get_api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        sys.exit("ERROR: SARVAM_API_KEY not set. Add it to .env or export it.")
    return key


def load_clips_meta() -> list[dict]:
    if not CLIPS_META_CSV.exists():
        sys.exit(f"ERROR: {CLIPS_META_CSV} not found. Run 03_preprocess_audio.py first.")
    with open(CLIPS_META_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_clips_meta(rows: list[dict]) -> None:
    with open(CLIPS_META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLIPS_META_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def submit_batch_job(clip_path: Path, language_code: str) -> str:
    """Submit a clip to Sarvam Batch STT in verbatim mode. Returns job_id."""
    api_key = get_api_key()
    headers = {"api-subscription-key": api_key}
    url = f"{SARVAM_BASE_URL}{SUBMIT_PATH}"

    with open(clip_path, "rb") as f:
        files = {"file": (clip_path.name, f, "audio/wav")}
        data = {
            "model": BATCH_MODEL,
            "language_code": language_code,
            "transcript_format": "verbatim",
        }
        resp = requests.post(url, headers=headers, files=files, data=data,
                             timeout=SUBMIT_TIMEOUT)

    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(
            f"Submit failed {resp.status_code}: {resp.text[:500]}\n"
            f"Verify endpoint at config/settings.yaml → sarvam.batch_submit_path"
        )

    body = resp.json()
    job_id = body.get("job_id") or body.get("request_id") or body.get("id")
    if not job_id:
        raise RuntimeError(f"No job_id in response: {body}")
    return str(job_id)


def poll_batch_job(job_id: str) -> str:
    """Poll until job completes. Returns transcript string."""
    api_key = get_api_key()
    headers = {"api-subscription-key": api_key}
    status_url = f"{SARVAM_BASE_URL}{STATUS_PATH}".replace("{job_id}", job_id)

    elapsed = 0
    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        resp = requests.get(status_url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"    Poll error {resp.status_code}, retrying...")
            continue

        body = resp.json()
        status = (body.get("status") or body.get("state") or "").lower()

        if status in ("completed", "done", "success"):
            result = body.get("output") or body.get("result") or body
            return (result.get("transcript") or result.get("text") or "").strip()
        elif status in ("failed", "error"):
            raise RuntimeError(f"Job {job_id} failed: {body}")

    raise TimeoutError(f"Job {job_id} did not complete within {MAX_WAIT}s")


def transcribe_clip(clip_path: Path, language_code: str) -> str:
    job_id = submit_batch_job(clip_path, language_code)
    return poll_batch_job(job_id)


def estimate_cost(rows: list[dict]) -> tuple[float, float]:
    total_s = sum(float(r.get("duration_s", 0) or 0) for r in rows)
    hours = total_s / 3600
    return hours, hours * COST_PER_HOUR_INR


def process_row(row: dict, rows: list[dict], dry_run: bool = False) -> bool:
    fn = row.get("clip_filename", "").strip()
    if not fn:
        return False

    clip_path = CLIPS_DIR / fn
    if not clip_path.exists():
        print(f"  [SKIP] Not found: {clip_path}")
        return False

    lang = row.get("language", "en").strip()
    lang_code = LANG_CODES.get(lang, "en-IN")
    dur = float(row.get("duration_s", 0) or 0)
    cost = (dur / 3600) * COST_PER_HOUR_INR

    print(f"  {fn} | {dur:.1f}s | {lang_code} | est. ₹{cost:.4f}")

    if dry_run:
        print("  [DRY RUN] Skipping actual API call.")
        return True

    try:
        transcript = transcribe_clip(clip_path, lang_code)
        row["transcript"] = transcript
        save_clips_meta(rows)  # write immediately after each clip
        preview = transcript[:80] + "..." if len(transcript) > 80 else transcript
        print(f"  [OK] {preview!r}")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe clips via Sarvam Batch ASR")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="Cost estimate + first untranscribed clip only")
    group.add_argument("--clip", metavar="FILENAME",
                       help="Transcribe one clip by filename")
    group.add_argument("--all", action="store_true",
                       help="Transcribe all clips missing a transcript")
    parser.add_argument("--redo", action="store_true",
                        help="Re-transcribe clips that already have a transcript (use with --all)")
    args = parser.parse_args()

    rows = load_clips_meta()

    if args.clip:
        pending = [r for r in rows if r.get("clip_filename") == args.clip]
        if not pending:
            sys.exit(f"Clip '{args.clip}' not found in {CLIPS_META_CSV}")
    elif args.all and args.redo:
        pending = rows
    else:
        pending = [r for r in rows if not r.get("transcript", "").strip()]

    hours, cost = estimate_cost(pending)
    print(f"Clips to transcribe: {len(pending)} | {hours:.3f} hrs | est. ₹{cost:.2f}")
    print()

    if args.dry_run:
        if not pending:
            print("No untranscribed clips found.")
            return
        print("=== DRY RUN: first clip only ===")
        process_row(pending[0], rows, dry_run=True)
        print(f"\nTo transcribe all: python scripts/04_transcribe.py --all")
        return

    if args.clip:
        process_row(pending[0], rows)
        return

    success = 0
    for i, row in enumerate(pending):
        print(f"[{i+1}/{len(pending)}]", end=" ")
        if process_row(row, rows):
            success += 1

    print(f"\nDone: {success}/{len(pending)} transcribed → {CLIPS_META_CSV}")


if __name__ == "__main__":
    main()

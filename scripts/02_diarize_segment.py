"""
Phase 2: Send downloaded audio to Sarvam's Batch STT API with diarization.
Uses the asynchronous Batch API (required for diarization and files > 30s).
Outputs raw diarization JSON per source file into data/segments/.

NOTE: Batch API endpoints are set in config/settings.yaml — verify against
https://docs.sarvam.ai before running if you get 404s.

IMPORTANT: This costs API credits (~₹45/hr of audio).
Always run --dry-run first to see the estimated cost.

After this phase, open each data/segments/*_diarization.json, look at the
speaker_summary field, identify which speaker_id is the target (the main speaker,
not a guest or interviewer), and fill in the `target_speaker_label` column in
metadata/source_log.csv before running Phase 3.

Usage:
    python scripts/02_diarize_segment.py --dry-run        # Cost estimate + first file only
    python scripts/02_diarize_segment.py --index 0        # Diarize one source
    python scripts/02_diarize_segment.py --all            # Diarize all (confirm cost first!)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
SOURCE_LOG_CSV = ROOT / "metadata" / "source_log.csv"
SEGMENTS_DIR = ROOT / "data" / "segments"
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


def get_api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        sys.exit("ERROR: SARVAM_API_KEY not set. Add it to .env or export it.")
    return key


def load_source_log() -> list[dict]:
    if not SOURCE_LOG_CSV.exists():
        sys.exit(f"ERROR: {SOURCE_LOG_CSV} not found. Run 01_download.py first.")
    with open(SOURCE_LOG_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_to_mp3(input_path: Path, output_path: Path) -> None:
    """Convert audio to mono MP3 for API upload (compact + widely supported)."""
    bitrate = SETTINGS["sarvam"]["upload_bitrate"]
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-vn", "-ac", "1", "-ab", bitrate, str(output_path)],
        capture_output=True, check=True,
    )


def submit_batch_job(mp3_path: Path, language_code: str) -> str:
    """Submit audio to Sarvam Batch STT API. Returns job_id."""
    api_key = get_api_key()
    headers = {"api-subscription-key": api_key}
    url = f"{SARVAM_BASE_URL}{SUBMIT_PATH}"

    print(f"  Submitting to Batch API: {url}")
    with open(mp3_path, "rb") as f:
        files = {"file": (mp3_path.name, f, "audio/mpeg")}
        data = {
            "model": BATCH_MODEL,
            "language_code": language_code,
            "with_diarization": "true",
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
    print(f"  Job submitted: {job_id}")
    return str(job_id)


def poll_batch_job(job_id: str) -> dict:
    """Poll until the job completes. Returns the result dict."""
    api_key = get_api_key()
    headers = {"api-subscription-key": api_key}
    status_url = f"{SARVAM_BASE_URL}{STATUS_PATH}".replace("{job_id}", job_id)

    elapsed = 0
    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        resp = requests.get(status_url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"  Poll error {resp.status_code}: {resp.text[:200]} — retrying...")
            continue

        body = resp.json()
        status = (body.get("status") or body.get("state") or "").lower()

        if status in ("completed", "done", "success"):
            return body.get("output") or body.get("result") or body
        elif status in ("failed", "error"):
            raise RuntimeError(f"Batch job {job_id} failed: {body}")

        mins = elapsed // 60
        print(f"  [{mins}m elapsed] Job {job_id}: {status or 'running'}...")

    raise TimeoutError(f"Job {job_id} did not complete within {MAX_WAIT}s")


def save_diarization(result: dict, source_row: dict, output_path: Path) -> None:
    """Parse API result, save enriched JSON, print speaker summary."""
    diarized = result.get("diarized_transcript", {})
    entries = diarized.get("entries", []) or diarized.get("segments", [])

    segments = [
        {
            "speaker_id": e.get("speaker_id", e.get("speaker", "UNKNOWN")),
            "start_time_s": e.get("start_time_seconds", e.get("start", 0)),
            "end_time_s": e.get("end_time_seconds", e.get("end", 0)),
            "text": e.get("transcript", e.get("text", "")),
        }
        for e in entries
    ]

    speaker_totals: dict[str, float] = {}
    for seg in segments:
        sid = seg["speaker_id"]
        dur = seg["end_time_s"] - seg["start_time_s"]
        speaker_totals[sid] = speaker_totals.get(sid, 0) + dur

    speaker_summary = {
        sid: round(s, 1)
        for sid, s in sorted(speaker_totals.items(), key=lambda x: -x[1])
    }

    output = {
        "source_index": source_row.get("source_index"),
        "speaker_name": source_row.get("speaker_name", ""),
        "language": source_row.get("language", ""),
        "video_url": source_row.get("video_url", ""),
        "local_file": source_row.get("local_file", ""),
        "speaker_summary": speaker_summary,
        "segments": segments,
        "api_response": result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def estimate_cost(rows: list[dict]) -> float:
    total_s = sum(float(r.get("duration_seconds", 0) or 0) for r in rows if r.get("local_file"))
    return (total_s / 3600) * COST_PER_HOUR_INR


def process_row(row: dict, dry_run: bool = False) -> bool:
    local_file = row.get("local_file", "").strip()
    if not local_file or not Path(local_file).exists():
        print(f"  [SKIP] No local file: {local_file!r}")
        return False

    source_path = Path(local_file)
    lang_code = LANG_CODES.get(row.get("language", "en").strip(), "en-IN")
    index = row.get("source_index", "?")

    output_json = SEGMENTS_DIR / f"{source_path.stem}_diarization.json"
    if output_json.exists():
        print(f"  [SKIP] Already diarized: {output_json.name}")
        return True

    dur = float(row.get("duration_seconds", 0) or 0)
    cost = (dur / 3600) * COST_PER_HOUR_INR
    print(f"  [{index}] {source_path.name} | {dur:.0f}s | est. ₹{cost:.2f}")

    if dry_run:
        print("  [DRY RUN] Skipping actual API call.")
        return True

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print("  Converting to MP3...")
        convert_to_mp3(source_path, tmp_path)

        job_id = submit_batch_job(tmp_path, lang_code)
        result = poll_batch_job(job_id)
        save_diarization(result, row, output_json)

        saved = json.loads(output_json.read_text(encoding="utf-8"))
        summary = saved.get("speaker_summary", {})
        print(f"  [OK] {output_json.name}")
        print(f"  Speaker totals (s): {summary}")
        print(f"  --> Identify the target speaker_id and fill 'target_speaker_label'")
        print(f"      in metadata/source_log.csv for index {index}.")
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diarize downloaded audio via Sarvam Batch API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="Cost estimate + first file only (no API call)")
    group.add_argument("--index", type=int,
                       help="Diarize one source by 0-based index")
    group.add_argument("--all", action="store_true",
                       help="Diarize all downloaded sources")
    args = parser.parse_args()

    rows = load_source_log()
    valid = [r for r in rows if r.get("local_file") and Path(r["local_file"]).exists()]
    if not valid:
        sys.exit("No downloaded files found. Run 01_download.py first.")

    total_hrs = sum(float(r.get("duration_seconds", 0) or 0) for r in valid) / 3600
    total_cost = estimate_cost(valid)
    print(f"{len(valid)} files | {total_hrs:.2f} hrs | full-batch est. ₹{total_cost:.2f} (~${total_cost/83:.2f} USD)")
    print()

    if args.dry_run:
        print("=== DRY RUN: first file only ===")
        process_row(valid[0], dry_run=True)
        print(f"\nTo process all: python scripts/02_diarize_segment.py --all")
        return

    if args.index is not None:
        target = next((r for r in rows if str(r.get("source_index")) == str(args.index)), None)
        if not target:
            sys.exit(f"No source at index {args.index}")
        process_row(target)
        return

    success = 0
    for row in valid:
        print(f"\n[{row.get('source_index')}] {row.get('speaker_name', 'Unknown')}")
        if process_row(row):
            success += 1
    print(f"\nDone: {success}/{len(valid)} diarized → {SEGMENTS_DIR}")


if __name__ == "__main__":
    main()

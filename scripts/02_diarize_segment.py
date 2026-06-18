"""
Phase 2: Diarize downloaded audio via Sarvam's Batch STT SDK (job lifecycle).

Job lifecycle per source file:
  1. create_job()      — register a new async job
  2. upload_files()    — upload the audio (webm/m4a/etc. natively supported)
  3. start()           — kick off processing
  4. wait_until_complete() — poll until done
  5. download_outputs()    — fetch the result JSON

Output per source: data/segments/{stem}_diarization.json
  - segments[]: {speaker_id, start_time_s, end_time_s, text}
  - speaker_summary: {SPEAKER_00: 200.1, SPEAKER_01: 40.3, ...}

After this phase: open each JSON, check speaker_summary (sorted by total time),
identify the target speaker_id, and fill in the `target_speaker_label` column in
metadata/source_log.csv before running Phase 3.

IMPORTANT: Run --dry-run first to see the cost estimate before spending credits.

Usage:
    python scripts/02_diarize_segment.py --dry-run        # Cost estimate, no API call
    python scripts/02_diarize_segment.py --index 0        # Single source
    python scripts/02_diarize_segment.py --all            # All (confirm cost first!)
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
SOURCE_LOG_CSV = ROOT / "metadata" / "source_log.csv"
SEGMENTS_DIR = ROOT / "data" / "segments"
SETTINGS = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())

BATCH_MODEL = SETTINGS["sarvam"]["batch_model"]
DIARIZATION_MODE = SETTINGS["sarvam"]["diarization_mode"]
COST_PER_HOUR_INR = SETTINGS["sarvam"]["cost_per_hour_inr"]
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


def parse_sdk_output(raw: dict) -> list[dict]:
    """Normalize SDK output into our segment format."""
    diarized = raw.get("diarized_transcript", {})
    entries = diarized.get("entries", []) or diarized.get("segments", [])
    return [
        {
            "speaker_id": e.get("speaker_id", "UNKNOWN"),
            "start_time_s": float(e.get("start_time_seconds", e.get("start", 0))),
            "end_time_s": float(e.get("end_time_seconds", e.get("end", 0))),
            "text": e.get("transcript", e.get("text", "")),
        }
        for e in entries
    ]


def build_speaker_summary(segments: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for seg in segments:
        sid = seg["speaker_id"]
        totals[sid] = totals.get(sid, 0) + (seg["end_time_s"] - seg["start_time_s"])
    return {sid: round(s, 1) for sid, s in sorted(totals.items(), key=lambda x: -x[1])}


def run_diarization_job(source_path: Path, lang_code: str) -> list[dict]:
    """Run one batch diarization job. Returns normalized segments list."""
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=get_api_key())

    print(f"  Creating job (model={BATCH_MODEL}, mode={DIARIZATION_MODE}, lang={lang_code})...")
    job = client.speech_to_text_job.create_job(
        model=BATCH_MODEL,
        mode=DIARIZATION_MODE,
        language_code=lang_code,
        with_diarization=True,
    )

    print(f"  Uploading {source_path.name}...")
    job.upload_files(file_paths=[str(source_path)])

    print("  Starting job...")
    job.start()

    print("  Waiting for completion (this may take several minutes)...")
    job.wait_until_complete()

    results = job.get_file_results()
    if results.get("failed"):
        for f in results["failed"]:
            raise RuntimeError(f"Job failed for {f['file_name']}: {f.get('error_message', '?')}")

    with tempfile.TemporaryDirectory() as tmpdir:
        job.download_outputs(output_dir=tmpdir)
        # SDK names output files by index: 0.json for the first uploaded file
        candidates = sorted(Path(tmpdir).glob("*.json"))
        if not candidates:
            raise RuntimeError(f"No JSON output found in download directory")
        raw = json.loads(candidates[0].read_text(encoding="utf-8"))

    return parse_sdk_output(raw)


def save_diarization(segments: list[dict], source_row: dict, output_path: Path) -> dict:
    summary = build_speaker_summary(segments)
    output = {
        "source_index": source_row.get("source_index"),
        "speaker_name": source_row.get("speaker_name", ""),
        "language": source_row.get("language", ""),
        "video_url": source_row.get("video_url", ""),
        "local_file": source_row.get("local_file", ""),
        "speaker_summary": summary,
        "segments": segments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def estimate_cost(rows: list[dict]) -> float:
    total_s = sum(float(r.get("duration_seconds", 0) or 0) for r in rows if r.get("local_file"))
    return (total_s / 3600) * COST_PER_HOUR_INR


def process_row(row: dict, dry_run: bool = False) -> bool:
    local_file = row.get("local_file", "").strip()
    if not local_file or not Path(local_file).exists():
        print(f"  [SKIP] No local file: {local_file!r}")
        return False

    source_path = Path(local_file)
    index = row.get("source_index", "?")
    lang_code = LANG_CODES.get(row.get("language", "en").strip(), "en-IN")
    output_json = SEGMENTS_DIR / f"{source_path.stem}_diarization.json"

    if output_json.exists():
        print(f"  [SKIP] Already diarized: {output_json.name}")
        return True

    dur = float(row.get("duration_seconds", 0) or 0)
    cost = (dur / 3600) * COST_PER_HOUR_INR
    print(f"  [{index}] {source_path.name} | {dur:.0f}s | est. ₹{cost:.2f}")

    if dry_run:
        print("  [DRY RUN] Skipping API call.")
        return True

    try:
        segments = run_diarization_job(source_path, lang_code)
        summary = save_diarization(segments, row, output_json)

        print(f"  [OK] {output_json.name} — {len(segments)} segments")
        print(f"  Speaker totals (s): {summary}")
        print(f"  --> Fill 'target_speaker_label' in source_log.csv for index {index}.")
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Diarize audio via Sarvam Batch SDK")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="Print cost estimate only — no API calls")
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
        print("=== DRY RUN: no API calls ===")
        process_row(valid[0], dry_run=True)
        print(f"\nTo diarize all: python scripts/02_diarize_segment.py --all")
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

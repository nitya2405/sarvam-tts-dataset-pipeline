"""
Phase 4: Transcribe clips via Sarvam Batch STT SDK in verbatim mode.

Uses the same job lifecycle as Phase 2 (create → upload → start → wait → download).
Clips are 45-60s so the REST API (< 30s limit) cannot be used.

Batches all pending clips into a single job per run for efficiency.
Writes transcripts back to clips_metadata.csv as soon as the job completes.

IMPORTANT: Run --dry-run first to see cost estimate.

Usage:
    python scripts/04_transcribe.py --dry-run                     # Cost estimate only
    python scripts/04_transcribe.py --clip en_news_000_0001.wav   # Single clip
    python scripts/04_transcribe.py --all                         # All without transcripts
    python scripts/04_transcribe.py --all --redo                  # Re-transcribe everything
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
CLIPS_META_CSV = ROOT / "metadata" / "clips_metadata.csv"
CLIPS_DIR = ROOT / "data" / "clips"
SETTINGS = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())

BATCH_MODEL = SETTINGS["sarvam"]["batch_model"]
TRANSCRIPTION_MODE = SETTINGS["sarvam"]["transcription_mode"]
COST_PER_HOUR_INR = SETTINGS["sarvam"]["cost_per_hour_inr"]
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


def estimate_cost(rows: list[dict]) -> tuple[float, float]:
    total_s = sum(float(r.get("duration_s", 0) or 0) for r in rows)
    hours = total_s / 3600
    return hours, hours * COST_PER_HOUR_INR


def transcribe_batch(clips: list[dict], rows: list[dict]) -> int:
    """
    Submit all clips as one batch job grouped by language.
    Clips within one language share a job; different languages need separate jobs
    because language_code is per-job in the SDK.
    Returns count of successfully transcribed clips.
    """
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=get_api_key())

    # Group by language so each job has a single language_code
    by_lang: dict[str, list[dict]] = {}
    for clip in clips:
        lang = clip.get("language", "en").strip()
        by_lang.setdefault(lang, []).append(clip)

    success_total = 0

    for lang, lang_clips in by_lang.items():
        lang_code = LANG_CODES.get(lang, "en-IN")
        clip_paths = [CLIPS_DIR / c["clip_filename"] for c in lang_clips]
        missing = [p for p in clip_paths if not p.exists()]
        if missing:
            print(f"  [WARN] Missing files, skipping: {[p.name for p in missing]}")
            clip_paths = [p for p in clip_paths if p.exists()]
            lang_clips = [c for c in lang_clips if (CLIPS_DIR / c["clip_filename"]).exists()]

        if not clip_paths:
            continue

        print(f"\n  Submitting {len(clip_paths)} {lang.upper()} clips "
              f"(model={BATCH_MODEL}, mode={TRANSCRIPTION_MODE}, lang={lang_code})...")

        job = client.speech_to_text_job.create_job(
            model=BATCH_MODEL,
            mode=TRANSCRIPTION_MODE,
            language_code=lang_code,
            with_diarization=False,
        )
        job.upload_files(file_paths=[str(p) for p in clip_paths])
        job.start()

        print(f"  Waiting for job to complete...")
        job.wait_until_complete()

        results = job.get_file_results()
        if results.get("failed"):
            for f in results["failed"]:
                print(f"  [FAIL] {f['file_name']}: {f.get('error_message', '?')}")

        with tempfile.TemporaryDirectory() as tmpdir:
            job.download_outputs(output_dir=tmpdir)
            output_files = sorted(Path(tmpdir).glob("*.json"))

            # SDK names outputs by upload index: 0.json, 1.json, ...
            for i, (clip_row, out_file) in enumerate(zip(lang_clips, output_files)):
                try:
                    raw = json.loads(out_file.read_text(encoding="utf-8"))
                    transcript = (
                        raw.get("transcript")
                        or raw.get("text")
                        or ""
                    ).strip()

                    # Write back to the matching row in rows
                    fn = clip_row["clip_filename"]
                    match = next((r for r in rows if r.get("clip_filename") == fn), None)
                    if match:
                        match["transcript"] = transcript
                        preview = transcript[:70] + "..." if len(transcript) > 70 else transcript
                        print(f"  [{i}] {fn}: {preview!r}")
                        success_total += 1
                except Exception as e:
                    print(f"  [ERROR] clip {i}: {e}")

        # Write after each language batch so progress is saved incrementally
        save_clips_meta(rows)

    return success_total


def transcribe_single(clip_row: dict, rows: list[dict]) -> bool:
    """Transcribe one clip as its own batch job."""
    fn = clip_row.get("clip_filename", "")
    clip_path = CLIPS_DIR / fn
    if not clip_path.exists():
        print(f"  [SKIP] Not found: {clip_path}")
        return False

    lang = clip_row.get("language", "en").strip()
    lang_code = LANG_CODES.get(lang, "en-IN")
    dur = float(clip_row.get("duration_s", 0) or 0)
    cost = (dur / 3600) * COST_PER_HOUR_INR
    print(f"  {fn} | {dur:.1f}s | {lang_code} | est. ₹{cost:.4f}")

    count = transcribe_batch([clip_row], rows)
    return count > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe clips via Sarvam Batch SDK")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="Cost estimate only — no API calls")
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
        print("=== DRY RUN: no API calls ===")
        first = pending[0]
        dur = float(first.get("duration_s", 0) or 0)
        print(f"  Would transcribe: {first['clip_filename']} ({dur:.1f}s)")
        print(f"\nTo transcribe all: python scripts/04_transcribe.py --all")
        return

    if args.clip:
        transcribe_single(pending[0], rows)
        return

    n = transcribe_batch(pending, rows)
    print(f"\nDone: {n}/{len(pending)} transcribed → {CLIPS_META_CSV}")


if __name__ == "__main__":
    main()

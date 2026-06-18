"""
Phase 3: Extract target-speaker segments, split into 45-60s clips, apply audio processing.

Prerequisites:
  - metadata/source_log.csv must have the 'target_speaker_label' column filled in
    (the speaker ID you identified from the diarization JSONs in Phase 2)
  - data/segments/*_diarization.json files must exist

Audio spec applied to every clip:
  - 24000 Hz, mono, 16-bit PCM WAV
  - High-pass filter at 80 Hz (removes rumble/hum)
  - Loudness normalization: -23 LUFS, -1 dBTP true-peak ceiling
  - Silence trim at edges with ~150-200 ms padding

QC flags written to clips_metadata.csv (flagged but not rejected):
  - clipping: any sample at or near full scale (>= 0.99 abs)
  - low_snr: integrated loudness of the raw clip below -35 LUFS (likely bad mic/noise)

Usage:
    python scripts/03_preprocess_audio.py --index 0     # Process single source
    python scripts/03_preprocess_audio.py --all         # Process all sources
    python scripts/03_preprocess_audio.py --all --overwrite  # Re-process already-done sources
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml
from dotenv import load_dotenv
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

ROOT = Path(__file__).parent.parent
SOURCE_LOG_CSV = ROOT / "metadata" / "source_log.csv"
SEGMENTS_DIR = ROOT / "data" / "segments"
CLIPS_DIR = ROOT / "data" / "clips"
CLIPS_META_CSV = ROOT / "metadata" / "clips_metadata.csv"
SETTINGS = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())

SR = SETTINGS["audio"]["sample_rate"]
MIN_CLIP_S = SETTINGS["audio"]["min_clip_duration_s"]
MAX_CLIP_S = SETTINGS["audio"]["max_clip_duration_s"]
TARGET_LUFS = SETTINGS["audio"]["loudness_target_lufs"]
TRUE_PEAK_DB = SETTINGS["audio"]["true_peak_ceiling_dbtp"]
HP_CUTOFF = SETTINGS["audio"]["highpass_cutoff_hz"]
PADDING_MS = SETTINGS["audio"]["silence_trim_padding_ms"]
MERGE_GAP_S = SETTINGS["audio"]["merge_gap_threshold_s"]
SILENCE_DB = SETTINGS["audio"]["silence_detection_threshold_db"]
SILENCE_MIN_MS = SETTINGS["audio"]["silence_min_len_ms"]

CLIPS_META_FIELDS = [
    "clip_filename", "language", "genre", "speaker_name",
    "source_index", "video_id", "source_channel", "source_url",
    "clip_start_time_s", "clip_end_time_s", "duration_s",
    "transcript", "primary_emotion", "secondary_emotion",
    "approved", "reviewed", "qc_flags",
]


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_as_mono_float(path: Path) -> tuple[np.ndarray, int]:
    """Load any audio file as float32 mono array at original SR, using pydub as fallback."""
    try:
        audio, sr = sf.read(str(path), always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(path))
        seg = seg.set_channels(1)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= float(1 << (seg.sample_width * 8 - 1))
        return samples, seg.frame_rate


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr)


def highpass_filter(audio: np.ndarray, sr: int, cutoff: int = 80) -> np.ndarray:
    nyquist = sr / 2.0
    sos = butter(4, cutoff / nyquist, btype="high", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def loudness_normalize(audio: np.ndarray, sr: int,
                        target_lufs: float = -23.0,
                        true_peak_db: float = -1.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(audio.astype(np.float64))
    if not np.isfinite(measured):
        return audio
    normalized = pyln.normalize.loudness(audio.astype(np.float64), measured, target_lufs)
    # Hard-limit to true peak ceiling
    ceiling_linear = 10 ** (true_peak_db / 20.0)
    peak = np.max(np.abs(normalized))
    if peak > ceiling_linear:
        normalized = normalized * (ceiling_linear / peak)
    return normalized.astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int, padding_ms: int = 175) -> np.ndarray:
    """Remove leading/trailing silence while keeping padding_ms of audio at each edge."""
    padding_samples = int(padding_ms * sr / 1000)
    frame_len = max(1, int(0.01 * sr))  # 10ms frames
    threshold_linear = 10 ** (SILENCE_DB / 20.0)

    def is_silent_frame(frame):
        return np.sqrt(np.mean(frame ** 2)) < threshold_linear

    # Find first non-silent frame
    start_frame = 0
    for i in range(0, len(audio), frame_len):
        if not is_silent_frame(audio[i: i + frame_len]):
            start_frame = i
            break

    # Find last non-silent frame
    end_frame = len(audio)
    for i in range(len(audio) - frame_len, -1, -frame_len):
        if not is_silent_frame(audio[i: i + frame_len]):
            end_frame = i + frame_len
            break

    start = max(0, start_frame - padding_samples)
    end = min(len(audio), end_frame + padding_samples)
    return audio[start:end]


def save_wav_16bit(audio: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_16 = np.clip(audio, -1.0, 1.0)
    audio_16 = (audio_16 * 32767).astype(np.int16)
    sf.write(str(path), audio_16, sr, subtype="PCM_16")


def qc_check(audio: np.ndarray, sr: int) -> list[str]:
    flags = []
    if np.any(np.abs(audio) >= 0.99):
        flags.append("clipping")
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(audio.astype(np.float64))
    if np.isfinite(lufs) and lufs < -35.0:
        flags.append("low_snr")
    return flags


# ---------------------------------------------------------------------------
# Silence boundary detection for clip splitting
# ---------------------------------------------------------------------------

def find_silence_regions(audio: np.ndarray, sr: int,
                          min_len_ms: int = 200,
                          thresh_db: float = -40.0) -> list[tuple[float, float]]:
    frame_len = max(1, int(0.01 * sr))
    threshold = 10 ** (thresh_db / 20.0)
    min_frames = max(1, int(min_len_ms / 10))

    regions = []
    in_silence = False
    silence_start = 0

    num_frames = len(audio) // frame_len
    for i in range(num_frames):
        frame = audio[i * frame_len: (i + 1) * frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        silent = rms < threshold

        if silent and not in_silence:
            in_silence = True
            silence_start = i
        elif not silent and in_silence:
            in_silence = False
            if i - silence_start >= min_frames:
                regions.append((silence_start * 0.01, i * 0.01))

    if in_silence and (num_frames - silence_start) >= min_frames:
        regions.append((silence_start * 0.01, num_frames * 0.01))

    return regions


def best_cut_point(silence_regions: list[tuple[float, float]],
                   min_s: float, max_s: float) -> float:
    target = (min_s + max_s) / 2
    in_window = [(s, e) for s, e in silence_regions if s >= min_s and e <= max_s]
    if in_window:
        best = min(in_window, key=lambda r: abs((r[0] + r[1]) / 2 - target))
        return (best[0] + best[1]) / 2
    before_max = [(s, e) for s, e in silence_regions if s < max_s]
    if before_max:
        return before_max[-1][0]
    return target


# ---------------------------------------------------------------------------
# Segment extraction and clip creation
# ---------------------------------------------------------------------------

def load_diarization(diarization_json: Path, target_speaker: str) -> list[dict]:
    data = json.loads(diarization_json.read_text(encoding="utf-8"))
    return [
        seg for seg in data.get("segments", [])
        if seg.get("speaker_id", "").strip().upper() == target_speaker.strip().upper()
    ]


def merge_adjacent_turns(turns: list[dict], gap_threshold_s: float) -> list[dict]:
    """Merge speaker turns that are separated by less than gap_threshold_s."""
    if not turns:
        return []
    turns = sorted(turns, key=lambda t: t["start_time_s"])
    merged = [dict(turns[0])]
    for turn in turns[1:]:
        gap = turn["start_time_s"] - merged[-1]["end_time_s"]
        if gap <= gap_threshold_s:
            merged[-1]["end_time_s"] = turn["end_time_s"]
            merged[-1]["text"] = merged[-1]["text"] + " " + turn.get("text", "")
        else:
            merged.append(dict(turn))
    return merged


def extract_audio_segment(audio: np.ndarray, sr: int,
                           start_s: float, end_s: float) -> np.ndarray:
    start = int(start_s * sr)
    end = int(end_s * sr)
    return audio[max(0, start): min(len(audio), end)]


def split_into_clips(segment_audio: np.ndarray, sr: int,
                     segment_start_s: float) -> list[dict]:
    """Split a long speaker segment into 45-60s clips at silence boundaries."""
    total_s = len(segment_audio) / sr
    clips = []
    offset = 0.0

    while offset < total_s:
        remaining = total_s - offset
        if remaining < MIN_CLIP_S:
            # Remainder is too short for a minimum-length clip — discard it.
            # (Attaching it to the previous clip bloats that clip well past MAX_CLIP_S.)
            break

        chunk_end_max = min(offset + MAX_CLIP_S, total_s)
        chunk_end_min = min(offset + MIN_CLIP_S, total_s)

        chunk = segment_audio[int(offset * sr): int(chunk_end_max * sr)]
        silence_regions = find_silence_regions(chunk, sr, SILENCE_MIN_MS, SILENCE_DB)

        cut_in_chunk = best_cut_point(
            silence_regions,
            min_s=chunk_end_min - offset,
            max_s=chunk_end_max - offset,
        )
        cut_abs = offset + cut_in_chunk

        clip_audio = segment_audio[int(offset * sr): int(cut_abs * sr)]
        if len(clip_audio) / sr >= MIN_CLIP_S - 1:
            clips.append({
                "audio": clip_audio,
                "start_s": segment_start_s + offset,
                "end_s": segment_start_s + cut_abs,
            })

        offset = cut_abs

    return clips


# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------

def load_clips_meta() -> list[dict]:
    if not CLIPS_META_CSV.exists():
        return []
    with open(CLIPS_META_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_clips_meta(rows: list[dict]) -> None:
    CLIPS_META_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CLIPS_META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLIPS_META_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_source(row: dict, meta_rows: list[dict], overwrite: bool = False) -> int:
    index = str(row.get("source_index", ""))
    local_file = row.get("local_file", "").strip()
    target_speaker = row.get("target_speaker_label", "").strip()

    if not local_file or not Path(local_file).exists():
        print(f"  [SKIP] local_file missing or not found: {local_file!r}")
        return 0

    if not target_speaker:
        print(f"  [SKIP] target_speaker_label not set in source_log.csv for index {index}")
        print(f"         Review data/segments/ diarization JSON, then fill in the column.")
        return 0

    source_path = Path(local_file)
    diarization_json = SEGMENTS_DIR / f"{source_path.stem}_diarization.json"
    if not diarization_json.exists():
        print(f"  [SKIP] Diarization JSON not found: {diarization_json}")
        return 0

    lang = row.get("language", "en").strip()
    genre = row.get("genre", "").strip()

    if not overwrite:
        existing = [r for r in meta_rows if str(r.get("source_index")) == index]
        if existing:
            print(f"  [SKIP] {len(existing)} clips already in metadata (use --overwrite to redo)")
            return 0

    if overwrite:
        # Delete all existing clips and metadata entries for this source before re-processing
        stale = [r for r in meta_rows if str(r.get("source_index")) == index]
        for r in stale:
            fn = r.get("clip_filename", "")
            if fn:
                (CLIPS_DIR / fn).unlink(missing_ok=True)
        meta_rows[:] = [r for r in meta_rows if str(r.get("source_index")) != index]

    print(f"  Loading audio: {source_path.name}")
    raw_audio, raw_sr = load_as_mono_float(source_path)

    turns = load_diarization(diarization_json, target_speaker)
    if not turns:
        print(f"  [WARN] No segments found for speaker '{target_speaker}' in {diarization_json.name}")
        return 0

    merged = merge_adjacent_turns(turns, MERGE_GAP_S)
    print(f"  Speaker '{target_speaker}': {len(turns)} turns -> {len(merged)} merged stretches")

    # Calculate the starting clip number ONCE before any clips are created.
    # Using this fixed offset + a local counter avoids double-incrementing.
    initial_clip_count = len([r for r in meta_rows if str(r.get("source_index")) == index])
    clip_count = 0

    for stretch in merged:
        stretch_start = stretch["start_time_s"]
        stretch_end = stretch["end_time_s"]
        stretch_dur = stretch_end - stretch_start

        if stretch_dur < MIN_CLIP_S:
            continue

        segment_audio = extract_audio_segment(raw_audio, raw_sr, stretch_start, stretch_end)
        segment_resampled = resample(segment_audio, raw_sr, SR)
        clips = split_into_clips(segment_resampled, SR, stretch_start)

        for clip_data in clips:
            clip_audio = clip_data["audio"]
            clip_start = clip_data["start_s"]
            clip_end = clip_data["end_s"]

            clip_filename = f"{lang}_{genre}_{int(index):03d}_{initial_clip_count + clip_count + 1:04d}.wav"
            clip_path = CLIPS_DIR / clip_filename

            # Apply audio processing pipeline
            processed = highpass_filter(clip_audio, SR, HP_CUTOFF)
            processed = loudness_normalize(processed, SR, TARGET_LUFS, TRUE_PEAK_DB)
            processed = trim_silence(processed, SR, PADDING_MS)

            if len(processed) / SR < MIN_CLIP_S * 0.8:
                print(f"  [WARN] Clip too short after processing ({len(processed)/SR:.1f}s), skipping")
                continue

            qc_flags = qc_check(processed, SR)
            save_wav_16bit(processed, SR, clip_path)

            actual_dur = len(processed) / SR
            meta_entry = {
                "clip_filename": clip_filename,
                "language": lang,
                "genre": genre,
                "speaker_name": row.get("speaker_name", ""),
                "source_index": index,
                "video_id": row.get("video_id", ""),
                "source_channel": row.get("channel_name", ""),
                "source_url": row.get("video_url", ""),
                "clip_start_time_s": round(clip_start, 3),
                "clip_end_time_s": round(clip_end, 3),
                "duration_s": round(actual_dur, 3),
                "transcript": "",
                "primary_emotion": row.get("target_emotion_primary", ""),
                "secondary_emotion": "",
                "approved": "",
                "reviewed": "false",
                "qc_flags": "|".join(qc_flags),
            }

            existing_idx = next(
                (i for i, r in enumerate(meta_rows) if r.get("clip_filename") == clip_filename),
                None,
            )
            if existing_idx is not None:
                meta_rows[existing_idx] = meta_entry
            else:
                meta_rows.append(meta_entry)

            flag_str = f" [FLAGS: {', '.join(qc_flags)}]" if qc_flags else ""
            print(f"  [CLIP] {clip_filename} | {actual_dur:.1f}s{flag_str}")
            clip_count += 1

    return clip_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract, split, and normalize audio clips")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int,
                       help="Process only the source at this 0-based index")
    group.add_argument("--all", action="store_true",
                       help="Process all sources in source_log.csv")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process clips that already exist")
    args = parser.parse_args()

    if not SOURCE_LOG_CSV.exists():
        sys.exit(f"ERROR: {SOURCE_LOG_CSV} not found. Run 01_download.py first.")

    with open(SOURCE_LOG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    meta_rows = load_clips_meta()
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    total_clips = 0
    sources_to_process = rows if args.all else [
        r for r in rows if str(r.get("source_index")) == str(args.index)
    ]

    if not sources_to_process:
        sys.exit(f"No matching sources found.")

    for row in sources_to_process:
        idx = row.get("source_index", "?")
        speaker = row.get("speaker_name", "Unknown")
        print(f"\n[{idx}] {speaker}")
        n = process_source(row, meta_rows, overwrite=args.overwrite)
        total_clips += n
        save_clips_meta(meta_rows)

    print(f"\nDone. {total_clips} new clips created.")
    print(f"Clips: {CLIPS_DIR}")
    print(f"Metadata: {CLIPS_META_CSV}")


if __name__ == "__main__":
    main()

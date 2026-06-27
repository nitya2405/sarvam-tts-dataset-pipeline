"""
Phase 7: Comprehensive evaluation of the approved dataset.

Outputs:
  - report/evaluation_report.md     — Markdown report
  - report/clip_metrics.csv         — Per-clip metrics table
  - evaluation_report.html          — Standalone visual dashboard (charts)

Usage:
    python scripts/07_evaluate_dataset.py
    python scripts/07_evaluate_dataset.py --all   # include unapproved clips too
"""

import argparse
import csv
import json as _json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
CLIPS_META_CSV = ROOT / "metadata" / "clips_metadata.csv"
CLIPS_DIR = ROOT / "data" / "clips"
REPORT_DIR = ROOT / "report"
REPORT_MD = REPORT_DIR / "evaluation_report.md"
CLIP_METRICS_CSV = REPORT_DIR / "clip_metrics.csv"
HTML_REPORT_PATH = ROOT / "evaluation_report.html"

SR = 24_000
SILENCE_THRESH_DB = -40.0
FRAME_S = 0.01
NOISE_PERCENTILE = 5

METRIC_FIELDS = [
    "clip_filename", "language", "genre", "speaker_name",
    "primary_emotion", "secondary_emotion", "approved", "qc_flags",
    "duration_s", "word_count", "words_per_min",
    "lufs", "lufs_delta",
    "true_peak_dbtp", "rms_dbfs", "snr_db",
    "spectral_centroid_hz", "silence_ratio",
    "transcript",
]


# ---------------------------------------------------------------------------
# Audio metrics
# ---------------------------------------------------------------------------

def _frame_rms(audio: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    val = meter.integrated_loudness(audio.astype(np.float64))
    return round(float(val), 2) if np.isfinite(val) else float("nan")


def measure_true_peak(audio: np.ndarray) -> float:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return -96.0
    return round(20 * np.log10(peak), 2)


def measure_rms_dbfs(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms == 0:
        return -96.0
    return round(20 * np.log10(rms), 2)


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    frame_len = max(1, int(FRAME_S * sr))
    frame_rms = _frame_rms(audio, frame_len)
    if len(frame_rms) == 0:
        return float("nan")
    signal_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    noise_rms = np.percentile(frame_rms, NOISE_PERCENTILE)
    if noise_rms == 0:
        return 60.0
    return round(float(20 * np.log10(signal_rms / noise_rms)), 1)


def measure_silence_ratio(audio: np.ndarray, sr: int) -> float:
    frame_len = max(1, int(FRAME_S * sr))
    frame_rms = _frame_rms(audio, frame_len)
    if len(frame_rms) == 0:
        return 0.0
    thresh = 10 ** (SILENCE_THRESH_DB / 20.0)
    return round(float(np.mean(frame_rms < thresh)), 3)


def measure_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    frame_len = 1024
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return float("nan")
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    window = np.hanning(frame_len)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    centroids = []
    for frame in frames:
        spectrum = np.abs(np.fft.rfft(frame * window))
        total = spectrum.sum()
        if total > 0:
            centroids.append(float(np.dot(freqs, spectrum) / total))
    return round(float(np.mean(centroids)), 1) if centroids else float("nan")


def analyze_clip(path: Path) -> dict:
    try:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            sr = SR
    except Exception as e:
        return {"error": str(e)}

    return {
        "lufs": measure_lufs(audio, sr),
        "true_peak_dbtp": measure_true_peak(audio),
        "rms_dbfs": measure_rms_dbfs(audio),
        "snr_db": estimate_snr(audio, sr),
        "spectral_centroid_hz": measure_spectral_centroid(audio, sr),
        "silence_ratio": measure_silence_ratio(audio, sr),
    }


# ---------------------------------------------------------------------------
# Transcript stats
# ---------------------------------------------------------------------------

def transcript_stats(text: str, duration_s: float) -> dict:
    words = text.split()
    wc = len(words)
    wpm = round(wc / (duration_s / 60), 1) if duration_s > 0 else 0.0
    return {"word_count": wc, "words_per_min": wpm}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(sum(vals) / len(vals), 2) if vals else float("nan")

def _std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.std(vals)), 2) if len(vals) > 1 else 0.0

def _pct(n, total):
    return f"{n} ({100 * n / total:.1f}%)" if total else "0"

def _af(pool, key):
    return [m[key] for m in pool if key in m and m[key] is not None
            and not (isinstance(m[key], float) and np.isnan(m[key]))]

def make_histogram_bins(values, n_bins=12):
    if not values:
        return [], []
    arr = np.array([v for v in values if v is not None and not np.isnan(float(v))])
    if len(arr) == 0:
        return [], []
    counts, edges = np.histogram(arr, bins=n_bins)
    labels = [f"{edges[i]:.1f}" for i in range(len(counts))]
    return labels, counts.tolist()


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def make_table(headers: list[str], rows: list[list]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    data_rows = [
        "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep] + data_rows)


def generate_markdown_report(metrics: list[dict], include_all: bool, repo_id: str) -> str:
    today = date.today().isoformat()
    approved = [m for m in metrics if m.get("approved", "").lower() == "true"]
    total = len(metrics)
    n_approved = len(approved)
    n_reviewed = sum(1 for m in metrics if m.get("reviewed", "").lower() == "true")

    pool = metrics if include_all else approved
    label = "all clips" if include_all else "approved clips"

    durations = [float(m["duration_s"]) for m in pool if m.get("duration_s")]
    total_min = round(sum(durations) / 60, 1)
    langs = sorted(set(m["language"] for m in pool))

    lang_stats = {}
    for lang in langs:
        clips = [m for m in pool if m["language"] == lang]
        dur_min = round(sum(float(m["duration_s"]) for m in clips) / 60, 1)
        lang_stats[lang.upper()] = {"clips": len(clips), "minutes": dur_min}

    genres = {}
    for m in pool:
        genres[m.get("genre", "?")] = genres.get(m.get("genre", "?"), 0) + 1
    genres = dict(sorted(genres.items(), key=lambda x: -x[1]))

    speakers = {}
    for m in pool:
        speakers[m.get("speaker_name", "?")] = speakers.get(m.get("speaker_name", "?"), 0) + 1
    speakers = dict(sorted(speakers.items(), key=lambda x: -x[1]))

    emotions = {}
    for m in pool:
        emotions[m.get("primary_emotion", "?")] = emotions.get(m.get("primary_emotion", "?"), 0) + 1
    emotions = dict(sorted(emotions.items(), key=lambda x: -x[1]))

    sec_emotions = {}
    for m in pool:
        e = m.get("secondary_emotion") or ""
        if e:
            sec_emotions[e] = sec_emotions.get(e, 0) + 1
    sec_emotions = dict(sorted(sec_emotions.items(), key=lambda x: -x[1]))

    emo_lang: dict[str, dict[str, int]] = {}
    for m in pool:
        lang = m.get("language", "?").upper()
        emo = m.get("primary_emotion", "?")
        emo_lang.setdefault(emo, {})
        emo_lang[emo][lang] = emo_lang[emo].get(lang, 0) + 1

    lufs_vals = _af(pool, "lufs")
    tp_vals = _af(pool, "true_peak_dbtp")
    rms_vals = _af(pool, "rms_dbfs")
    snr_vals = _af(pool, "snr_db")
    sc_vals = _af(pool, "spectral_centroid_hz")
    sr_vals = _af(pool, "silence_ratio")
    wc_vals = _af(pool, "word_count")
    wpm_vals = _af(pool, "words_per_min")

    n_measured = len(lufs_vals)
    lufs_ok = sum(1 for v in lufs_vals if abs(v - (-23.0)) <= 1.0)
    tp_ok = sum(1 for v in tp_vals if v <= -1.0)
    snr_good = sum(1 for v in snr_vals if v >= 25.0)
    snr_ok = sum(1 for v in snr_vals if 15.0 <= v < 25.0)
    snr_poor = sum(1 for v in snr_vals if v < 15.0)
    flagged = [m for m in pool if m.get("qc_flags", "").strip()]

    lines = [
        f"# Dataset Evaluation Report",
        f"",
        f"Generated: {today}  ",
        f"Repo: `{repo_id}`  ",
        f"Scope: **{label}**",
        f"",
        f"---",
        f"",
        f"## 1. Collection Summary",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total clips in metadata | {total} |",
        f"| Reviewed | {n_reviewed} ({100*n_reviewed//max(total,1)}%) |",
        f"| Approved | {n_approved} ({100*n_approved//max(total,1)}%) |",
        f"| Clips evaluated ({label}) | {len(pool)} |",
        f"| Total duration | {total_min} min |",
        f"| Unique speakers | {len(speakers)} |",
        f"| Unique genres | {len(genres)} |",
        f"| Languages | {', '.join(langs)} |",
        f"",
        f"### Duration stats ({label})",
        f"",
        f"| Stat | Value |",
        f"|---|---|",
        f"| Min | {min(durations):.1f} s |" if durations else "| Min | — |",
        f"| Max | {max(durations):.1f} s |" if durations else "| Max | — |",
        f"| Mean | {_mean(durations):.1f} s |" if durations else "| Mean | — |",
        f"| Std | {_std(durations):.1f} s |" if durations else "| Std | — |",
        f"",
        f"---",
        f"",
        f"## 2. Language & Genre Distribution",
        f"",
    ]

    lang_rows = [[lang, v["clips"], f"{v['minutes']} min"] for lang, v in lang_stats.items()]
    lines.append(make_table(["Language", "Clips", "Duration"], lang_rows))
    lines.append("")

    genre_rows = [[g, cnt, _pct(cnt, len(pool))] for g, cnt in genres.items()]
    lines.append(make_table(["Genre", "Clips", "Share"], genre_rows))
    lines += ["", "---", "", "## 3. Speaker Distribution", ""]
    lines.append(make_table(["Speaker", "Clips"], [[sp, cnt] for sp, cnt in speakers.items()]))
    lines += ["", "---", "", "## 4. Emotion Distribution", "", "### Primary emotion", ""]
    lines.append(make_table(["Emotion", "Clips", "Share"],
                             [[emo, cnt, _pct(cnt, len(pool))] for emo, cnt in emotions.items()]))
    lines.append("")

    if sec_emotions:
        lines += ["### Secondary emotion", ""]
        lines.append(make_table(["Emotion", "Clips", "Share"],
                                 [[emo, cnt, _pct(cnt, len(pool))] for emo, cnt in sec_emotions.items()]))
        lines.append("")

    lines += ["### Primary emotion × language", ""]
    all_emo_langs = sorted(set().union(*[set(v.keys()) for v in emo_lang.values()]))
    lines.append(make_table(
        ["Emotion"] + all_emo_langs,
        [[emo] + [emo_lang.get(emo, {}).get(l, 0) for l in all_emo_langs]
         for emo in sorted(emo_lang.keys())]
    ))
    lines += ["", "---", "", "## 5. Audio Quality Metrics", ""]

    if n_measured == 0:
        lines.append("_No audio files could be measured._")
    else:
        lines.append(f"Measured: **{n_measured}** clips\n")
        lines.append("### Aggregate audio stats\n")
        lines.append(make_table(
            ["Metric", "Mean", "Std", "Min", "Max"],
            [
                ["LUFS (target −23)", _mean(lufs_vals), _std(lufs_vals),
                 round(min(lufs_vals), 2), round(max(lufs_vals), 2)],
                ["True peak dBTP (ceiling −1)", _mean(tp_vals), _std(tp_vals),
                 round(min(tp_vals), 2), round(max(tp_vals), 2)],
                ["RMS dBFS", _mean(rms_vals), _std(rms_vals),
                 round(min(rms_vals), 2), round(max(rms_vals), 2)],
                ["Estimated SNR dB", _mean(snr_vals), _std(snr_vals),
                 round(min(snr_vals), 2), round(max(snr_vals), 2)],
                ["Spectral centroid Hz", _mean(sc_vals), _std(sc_vals),
                 round(min(sc_vals), 1), round(max(sc_vals), 1)],
                ["Silence ratio", _mean(sr_vals), _std(sr_vals),
                 round(min(sr_vals), 3), round(max(sr_vals), 3)],
            ]
        ))
        lines.append("")
        lines.append("### Loudness conformance\n")
        lines.append(make_table(
            ["Check", "Pass", "Fail"],
            [
                ["LUFS within ±1 dB of −23", _pct(lufs_ok, n_measured),
                 _pct(n_measured - lufs_ok, n_measured)],
                ["True peak ≤ −1 dBTP", _pct(tp_ok, n_measured),
                 _pct(n_measured - tp_ok, n_measured)],
            ]
        ))
        lines.append("")
        lines.append("### SNR quality bands\n")
        lines.append(make_table(
            ["Band", "Count"],
            [
                ["Good (≥25 dB)", _pct(snr_good, n_measured)],
                ["OK (15–25 dB)", _pct(snr_ok, n_measured)],
                ["Poor (<15 dB)", _pct(snr_poor, n_measured)],
            ]
        ))
        lines.append("")

        LUFS_WARN = 2.0
        outliers = [m for m in pool if "lufs" in m and not np.isnan(m.get("lufs", float("nan")))
                    and abs(m["lufs"] - (-23.0)) > LUFS_WARN]
        if outliers:
            lines.append(f"### Loudness outliers (|LUFS − (−23)| > {LUFS_WARN} dB)\n")
            lines.append(make_table(
                ["Clip", "LUFS", "Delta", "Emotion", "Genre"],
                [[m["clip_filename"], m["lufs"], round(m["lufs"] - (-23.0), 2),
                  m.get("primary_emotion", ""), m.get("genre", "")]
                 for m in sorted(outliers, key=lambda x: abs(x["lufs"] - (-23.0)), reverse=True)]
            ))
            lines.append("")

        if flagged:
            lines.append("### QC-flagged clips\n")
            lines.append(make_table(
                ["Clip", "Flags", "Approved", "Emotion"],
                [[m["clip_filename"], m["qc_flags"], m.get("approved", ""),
                  m.get("primary_emotion", "")] for m in flagged]
            ))
            lines.append("")

    lines += ["---", "", "## 6. Transcript / Speech Rate", ""]
    if wc_vals:
        lines.append(make_table(
            ["Stat", "Word count", "Words per min"],
            [
                ["Mean", _mean(wc_vals), _mean(wpm_vals)],
                ["Std", _std(wc_vals), _std(wpm_vals)],
                ["Min", min(wc_vals), min(wpm_vals)],
                ["Max", max(wc_vals), max(wpm_vals)],
            ]
        ))
        lines.append("")
        lines.append("### Speech rate by genre\n")
        genre_wpm_rows = []
        for g in sorted(genres.keys()):
            gclips = [m for m in pool if m.get("genre") == g and m.get("words_per_min") is not None]
            wpm_g = [m["words_per_min"] for m in gclips]
            genre_wpm_rows.append([g, len(gclips), _mean(wpm_g), _std(wpm_g)])
        lines.append(make_table(["Genre", "N", "Mean WPM", "Std WPM"], genre_wpm_rows))
        lines.append("")

        fast = sorted([m for m in pool if m.get("words_per_min", 0) > 180],
                      key=lambda x: -x["words_per_min"])
        if fast:
            lines.append("### Unusually fast speech (>180 WPM)\n")
            lines.append(make_table(
                ["Clip", "WPM", "Genre", "Speaker"],
                [[m["clip_filename"], m["words_per_min"], m.get("genre", ""), m.get("speaker_name", "")]
                 for m in fast]
            ))
            lines.append("")

    lines += ["---", "", "## 7. Dataset Readiness Assessment", ""]
    issues, suggestions = [], []
    for lang, v in lang_stats.items():
        if v["minutes"] < 20:
            issues.append(f"Language **{lang}** has only {v['minutes']} min (< 20 min recommended)")
    for emo, cnt in emotions.items():
        if cnt < 3:
            issues.append(f"Emotion `{emo}` has only {cnt} clip(s) — very few examples")
    if n_measured > 0 and snr_poor > 0:
        issues.append(f"{snr_poor} clip(s) with poor SNR (<15 dB)")
    if n_measured > 0 and (n_measured - tp_ok) > 0:
        issues.append(f"{n_measured - tp_ok} clip(s) exceed −1 dBTP ceiling")

    if not issues and not suggestions:
        lines.append("**No significant issues found.** The dataset looks balanced and clean.")
        lines.append("")
    else:
        if issues:
            lines += ["### Issues", ""] + [f"- {i}" for i in issues] + [""]
        if suggestions:
            lines += ["### Suggestions", ""] + [f"- {s}" for s in suggestions] + [""]

    lines += ["---", "", f"_Report generated by `scripts/07_evaluate_dataset.py` on {today}._"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sarvam TTS — Dataset Evaluation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f1f5f9;
  --card:#ffffff;
  --border:#e2e8f0;
  --text:#0f172a;
  --muted:#64748b;
  --accent:#3b82f6;
  --green:#16a34a;
  --green-bg:#f0fdf4;
  --green-bd:#86efac;
  --amber:#b45309;
  --amber-bg:#fffbeb;
  --amber-bd:#fcd34d;
}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:13px;line-height:1.6;min-height:100vh}

/* ── Header ── */
.hdr{
  background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#1e40af 100%);
  color:#fff;padding:48px 56px 40px;
}
.hdr-eyebrow{font-size:11px;font-weight:500;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.45);margin-bottom:10px}
.hdr-title{font-size:28px;font-weight:700;letter-spacing:-.5px;line-height:1.2}
.hdr-sub{font-size:13px;color:rgba(255,255,255,.5);margin-top:6px}
.stat-row{display:flex;flex-wrap:wrap;gap:12px;margin-top:32px}
.stat{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:14px 20px;min-width:110px}
.stat-val{font-size:22px;font-weight:700;line-height:1}
.stat-val.good{color:#4ade80}
.stat-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:rgba(255,255,255,.45);margin-top:5px}

/* ── Badges ── */
.badge-row{display:flex;flex-wrap:wrap;gap:8px;background:#fff;padding:14px 56px;border-bottom:1px solid var(--border)}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;font-weight:500;padding:5px 12px;border-radius:20px}
.badge.ok{background:var(--green-bg);color:var(--green);border:1px solid var(--green-bd)}
.badge.warn{background:var(--amber-bg);color:var(--amber);border:1px solid var(--amber-bd)}
.badge-icon{font-size:12px}

/* ── Layout ── */
.main{max-width:1320px;margin:0 auto;padding:36px 56px 48px}
.sec-head{font-size:15px;font-weight:700;color:var(--text);margin:40px 0 2px}
.sec-head:first-of-type{margin-top:0}
.sec-sub{font-size:12px;color:var(--muted);margin-bottom:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid.three{grid-template-columns:1fr 1fr 1fr}
.full{grid-column:1/-1}
.card{background:var(--card);border-radius:12px;border:1px solid var(--border);box-shadow:0 1px 2px rgba(0,0,0,.04),0 4px 12px rgba(0,0,0,.03);padding:22px 26px}
.card-label{font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.9px;color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:6px}
.card-label span{font-size:14px}
.wrap{position:relative}

/* ── Language special card ── */
.lang-card{display:flex;align-items:center;gap:24px}
.lang-donut-wrap{flex:0 0 200px;height:200px;position:relative}
.lang-info{flex:1;display:flex;flex-direction:column;gap:10px}
.lang-row2{display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:8px;background:var(--bg)}
.lang-dot{width:11px;height:11px;border-radius:50%;flex-shrink:0}
.lang-name{font-weight:700;font-size:15px;letter-spacing:-.3px}
.lang-detail{font-size:12px;color:var(--muted);margin-top:1px}
.lang-detail strong{color:var(--text)}

/* ── Quality metrics row ── */
.qm-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.qm-card{background:var(--card);border-radius:10px;border:1px solid var(--border);padding:14px 18px;text-align:center}
.qm-val{font-size:20px;font-weight:700;color:var(--accent)}
.qm-val.great{color:#16a34a}
.qm-val.warn{color:#d97706}
.qm-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-top:4px}

/* ── Footer ── */
.footer{text-align:center;padding:28px;color:var(--muted);font-size:11px;border-top:1px solid var(--border);margin-top:12px}
.footer a{color:var(--accent);text-decoration:none}
.footer a:hover{text-decoration:underline}

@media(max-width:900px){
  .grid,.grid.three{grid-template-columns:1fr}
  .full{grid-column:1}
  .hdr,.main,.badge-row{padding-left:24px;padding-right:24px}
  .qm-grid{grid-template-columns:1fr 1fr}
  .lang-card{flex-direction:column}
}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-eyebrow">Dataset Evaluation Report</div>
  <div class="hdr-title" id="hdr-title">Sarvam TTS Dataset</div>
  <div class="hdr-sub" id="hdr-sub"></div>
  <div class="stat-row" id="stat-row"></div>
</div>

<div class="badge-row" id="badge-row"></div>

<div class="main">

  <!-- Audio quality metrics row -->
  <div class="qm-grid" id="qm-grid"></div>

  <!-- Section 1: Composition -->
  <div class="sec-head">Composition &amp; Coverage</div>
  <div class="sec-sub">Language balance, emotional range, and genre diversity across the dataset.</div>
  <div class="grid" style="margin-bottom:16px">
    <div class="card">
      <div class="card-label"><span>🌐</span>Language Balance</div>
      <div class="lang-card">
        <div class="lang-donut-wrap"><canvas id="c-lang"></canvas></div>
        <div class="lang-info" id="lang-info"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-label"><span>🎭</span>Primary Emotion Distribution</div>
      <div class="wrap" style="height:220px"><canvas id="c-emotion"></canvas></div>
    </div>
  </div>

  <!-- Section 2: Diversity -->
  <div class="sec-head">Content Diversity</div>
  <div class="sec-sub">Genre coverage and speaker variety ensure the dataset generalises well.</div>
  <div class="grid" style="margin-bottom:16px">
    <div class="card">
      <div class="card-label"><span>🎙</span>Genre Distribution</div>
      <div class="wrap" style="height:240px"><canvas id="c-genre"></canvas></div>
    </div>
    <div class="card">
      <div class="card-label"><span>👤</span>Speaker Distribution</div>
      <div class="wrap" style="height:400px"><canvas id="c-speaker"></canvas></div>
    </div>
  </div>

  <!-- Section 3: Audio Quality -->
  <div class="sec-head">Audio Quality</div>
  <div class="sec-sub">LUFS deviation from the −23 target, SNR distribution, and clip length consistency.</div>
  <div class="grid" style="margin-bottom:16px">
    <div class="card">
      <div class="card-label"><span>📊</span>LUFS Deviation from −23 Target</div>
      <div class="wrap" style="height:190px"><canvas id="c-lufs"></canvas></div>
    </div>
    <div class="card">
      <div class="card-label"><span>📡</span>Signal-to-Noise Ratio (dB)</div>
      <div class="wrap" style="height:190px"><canvas id="c-snr"></canvas></div>
    </div>
    <div class="card">
      <div class="card-label"><span>⏱</span>Clip Duration (seconds)</div>
      <div class="wrap" style="height:190px"><canvas id="c-dur"></canvas></div>
    </div>
    <div class="card">
      <div class="card-label"><span>🔊</span>Spectral Centroid (Hz)</div>
      <div class="wrap" style="height:190px"><canvas id="c-sc"></canvas></div>
    </div>
  </div>

  <!-- Section 4: Speech Rate -->
  <div class="sec-head">Speech Rate</div>
  <div class="sec-sub">Words per minute distribution and secondary emotion tagging.</div>
  <div class="grid">
    <div class="card">
      <div class="card-label"><span>💬</span>Speech Rate Distribution (WPM)</div>
      <div class="wrap" style="height:190px"><canvas id="c-wpm"></canvas></div>
    </div>
    <div class="card">
      <div class="card-label"><span>🌀</span>Secondary Emotion Distribution</div>
      <div class="wrap" style="height:220px"><canvas id="c-sec-emotion"></canvas></div>
    </div>
  </div>

</div>

<div class="footer" id="footer"></div>

<script>
const D = ___CHART_DATA___;

// ── Chart.js global defaults ──────────────────────────────────────────────
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.color = '#64748b';
Chart.defaults.plugins.legend.display = false;
Chart.defaults.plugins.tooltip.backgroundColor = '#0f172a';
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
Chart.defaults.plugins.tooltip.bodyColor = '#94a3b8';
Chart.defaults.plugins.tooltip.titleFont = { size: 12, weight: '600' };
Chart.defaults.plugins.tooltip.bodyFont = { size: 11 };

const PALETTE = [
  '#3b82f6','#8b5cf6','#ec4899','#06b6d4',
  '#10b981','#f59e0b','#ef4444','#6366f1',
  '#14b8a6','#f97316','#84cc16','#e879f9',
];

const SCALE = {
  grid: { color: '#f1f5f9', drawBorder: false },
  ticks: { color: '#94a3b8', font: { size: 10 } },
  border: { display: false },
};

function hbar(id, labels, values, colors) {
  const bg = colors || labels.map((_, i) => PALETTE[i % PALETTE.length]);
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: bg, borderRadius: 5, borderWidth: 0 }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ...SCALE, beginAtZero: true, grid: { color: '#f1f5f9' } },
        y: { ...SCALE, ticks: { ...SCALE.ticks, font: { size: 11 } } },
      },
    },
  });
}

function hist(id, labels, values, color, xLabel) {
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: color + 'cc',
        borderColor: color,
        borderWidth: 1.5,
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ...SCALE, title: { display: !!xLabel, text: xLabel, color: '#94a3b8', font: { size: 10 } } },
        y: { ...SCALE, beginAtZero: true, ticks: { precision: 0 } },
      },
      categoryPercentage: 0.95, barPercentage: 0.9,
    },
  });
}

function doughnut(id, labels, values, colors) {
  const bg = colors || labels.map((_, i) => PALETTE[i % PALETTE.length]);
  return new Chart(document.getElementById(id), {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: bg, borderWidth: 2, borderColor: '#fff', hoverOffset: 4 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed} clips` } },
      },
    },
  });
}

// ── Populate header ───────────────────────────────────────────────────────
const m = D.meta;
document.getElementById('hdr-title').textContent = 'Sarvam TTS Dataset';
document.getElementById('hdr-sub').textContent =
  m.repo_id + '  ·  Generated ' + m.today + '  ·  Scope: ' + m.label;

const statDefs = [
  { val: m.n_clips,                   lbl: 'Approved Clips',    cls: '' },
  { val: m.total_min + ' min',        lbl: 'Total Duration',    cls: '' },
  { val: m.n_speakers,                lbl: 'Unique Speakers',   cls: '' },
  { val: m.n_genres,                  lbl: 'Genres',            cls: '' },
  { val: m.mean_snr + ' dB',         lbl: 'Mean SNR',          cls: 'good' },
  { val: m.lufs_ok_pct + '%',        lbl: 'LUFS Conformance',  cls: m.lufs_ok_pct >= 95 ? 'good' : '' },
];
const statRow = document.getElementById('stat-row');
statDefs.forEach(s => {
  statRow.innerHTML += `<div class="stat"><div class="stat-val ${s.cls}">${s.val}</div><div class="stat-lbl">${s.lbl}</div></div>`;
});

// ── Quality metrics row ───────────────────────────────────────────────────
const qmDefs = [
  { val: m.lufs_ok_pct + '%',   lbl: 'LUFS ±1 dB Conformance', cls: m.lufs_ok_pct >= 95 ? 'great' : 'warn' },
  { val: m.tp_ok_pct + '%',     lbl: 'True Peak ≤ −1 dBTP',   cls: m.tp_ok_pct >= 100 ? 'great' : 'warn' },
  { val: m.mean_lufs,            lbl: 'Mean LUFS (target −23)', cls: '' },
  { val: m.n_poor_snr === 0 ? '0 clips' : m.n_poor_snr + ' clips', lbl: 'Poor SNR (<15 dB)', cls: m.n_poor_snr === 0 ? 'great' : 'warn' },
];
const qmGrid = document.getElementById('qm-grid');
qmDefs.forEach(q => {
  qmGrid.innerHTML += `<div class="qm-card"><div class="qm-val ${q.cls}">${q.val}</div><div class="qm-lbl">${q.lbl}</div></div>`;
});

// ── Badges ────────────────────────────────────────────────────────────────
const badgeDefs = [
  { ok: m.tp_ok_pct >= 100,            text: '100% True Peak Compliance' },
  { ok: m.n_poor_snr === 0,            text: 'Zero Poor-SNR Clips' },
  { ok: m.lufs_ok_pct >= 95,           text: m.lufs_ok_pct + '% LUFS Conformance' },
  { ok: m.n_speakers >= 10,            text: m.n_speakers + ' Unique Speakers' },
  { ok: m.n_genres >= 7,               text: m.n_genres + ' Genres Covered' },
  { ok: m.lang_balanced,               text: 'Balanced EN / HI Split' },
];
const badgeRow = document.getElementById('badge-row');
badgeDefs.forEach(b => {
  const cls = b.ok ? 'ok' : 'warn';
  const icon = b.ok ? '✓' : '⚠';
  badgeRow.innerHTML += `<span class="badge ${cls}"><span class="badge-icon">${icon}</span>${b.text}</span>`;
});

// ── Language doughnut + info ──────────────────────────────────────────────
const langColors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'];
doughnut('c-lang', D.language.labels, D.language.clips, langColors);
const langInfo = document.getElementById('lang-info');
D.language.labels.forEach((lang, i) => {
  langInfo.innerHTML += `
    <div class="lang-row2">
      <div class="lang-dot" style="background:${langColors[i % langColors.length]}"></div>
      <div>
        <div class="lang-name">${lang}</div>
        <div class="lang-detail"><strong>${D.language.clips[i]}</strong> clips &nbsp;·&nbsp; <strong>${D.language.minutes[i]}</strong> min</div>
      </div>
    </div>`;
});

// ── Emotion charts ────────────────────────────────────────────────────────
hbar('c-emotion', D.emotion.labels, D.emotion.values);
hbar('c-sec-emotion', D.sec_emotion.labels, D.sec_emotion.values);

// ── Genre & Speaker ───────────────────────────────────────────────────────
hbar('c-genre', D.genre.labels, D.genre.values, D.genre.labels.map((_, i) => PALETTE[i % PALETTE.length]));
hbar('c-speaker', D.speaker.labels, D.speaker.values, '#3b82f6');

// ── Histograms ────────────────────────────────────────────────────────────
hist('c-lufs', D.lufs_hist.labels, D.lufs_hist.values, '#3b82f6', 'LUFS Δ from −23');
hist('c-snr',  D.snr_hist.labels,  D.snr_hist.values,  '#10b981', 'SNR (dB)');
hist('c-dur',  D.dur_hist.labels,  D.dur_hist.values,  '#8b5cf6', 'Duration (s)');
hist('c-wpm',  D.wpm_hist.labels,  D.wpm_hist.values,  '#f59e0b', 'Words per minute');
hist('c-sc',   D.sc_hist.labels,   D.sc_hist.values,   '#06b6d4', 'Centroid (Hz)');

// ── Footer ────────────────────────────────────────────────────────────────
document.getElementById('footer').innerHTML =
  'Generated by <code>scripts/07_evaluate_dataset.py</code> on ' + m.today +
  ' &nbsp;·&nbsp; <a href="https://huggingface.co/datasets/' + m.repo_id + '" target="_blank">View on Hugging Face ↗</a>';
</script>
</body>
</html>"""


def generate_html_report(metrics: list[dict], include_all: bool, repo_id: str) -> str:
    pool = metrics if include_all else [m for m in metrics if m.get("approved", "").lower() == "true"]
    today = date.today().isoformat()
    label = "all clips" if include_all else "approved clips"

    durations = [float(m["duration_s"]) for m in pool if m.get("duration_s")]
    total_min = round(sum(durations) / 60, 1)
    n_speakers = len(set(m.get("speaker_name", "") for m in pool))
    langs = sorted(set(m.get("language", "?") for m in pool))
    genres = {}
    for m in pool:
        genres[m.get("genre", "?")] = genres.get(m.get("genre", "?"), 0) + 1

    lufs_vals = _af(pool, "lufs")
    snr_vals = _af(pool, "snr_db")
    tp_vals = _af(pool, "true_peak_dbtp")
    wpm_vals = _af(pool, "words_per_min")
    sc_vals = _af(pool, "spectral_centroid_hz")
    dur_vals = [float(m["duration_s"]) for m in pool if m.get("duration_s")]

    lufs_ok_pct = round(100 * sum(1 for v in lufs_vals if abs(v - (-23.0)) <= 1.0) / max(len(lufs_vals), 1), 1)
    tp_ok_pct = round(100 * sum(1 for v in tp_vals if v <= -1.0) / max(len(tp_vals), 1), 1)
    n_poor_snr = sum(1 for v in snr_vals if v < 15.0)

    lang_clips = [sum(1 for m in pool if m.get("language") == l) for l in langs]
    lang_mins = [round(sum(float(m["duration_s"]) for m in pool if m.get("language") == l) / 60, 1) for l in langs]
    lang_balanced = len(lang_clips) >= 2 and max(lang_clips) / max(min(lang_clips), 1) < 1.25

    emotions = {}
    for m in pool:
        emotions[m.get("primary_emotion", "?")] = emotions.get(m.get("primary_emotion", "?"), 0) + 1
    emotions = dict(sorted(emotions.items(), key=lambda x: -x[1]))

    sec_emotions = {}
    for m in pool:
        e = m.get("secondary_emotion") or ""
        if e:
            sec_emotions[e] = sec_emotions.get(e, 0) + 1
    sec_emotions = dict(sorted(sec_emotions.items(), key=lambda x: -x[1]))

    speakers = {}
    for m in pool:
        speakers[m.get("speaker_name", "?")] = speakers.get(m.get("speaker_name", "?"), 0) + 1
    speakers = dict(sorted(speakers.items(), key=lambda x: -x[1]))

    genres_sorted = dict(sorted(genres.items(), key=lambda x: -x[1]))

    lufs_delta = [round(v - (-23.0), 3) for v in lufs_vals]
    dur_labels, dur_counts = make_histogram_bins(dur_vals, 10)
    wpm_labels, wpm_counts = make_histogram_bins(wpm_vals, 12)
    delta_labels, delta_counts = make_histogram_bins(lufs_delta, 10)
    snr_labels, snr_counts = make_histogram_bins(snr_vals, 12)
    sc_labels, sc_counts = make_histogram_bins(sc_vals, 10)

    chart_data = {
        "meta": {
            "n_clips": len(pool),
            "total_min": total_min,
            "n_speakers": n_speakers,
            "n_genres": len(genres),
            "mean_snr": _mean(snr_vals),
            "mean_lufs": _mean(lufs_vals),
            "lufs_ok_pct": lufs_ok_pct,
            "tp_ok_pct": tp_ok_pct,
            "n_poor_snr": n_poor_snr,
            "lang_balanced": lang_balanced,
            "today": today,
            "repo_id": repo_id,
            "label": label,
        },
        "language": {
            "labels": [l.upper() for l in langs],
            "clips": lang_clips,
            "minutes": lang_mins,
        },
        "emotion": {
            "labels": list(emotions.keys()),
            "values": list(emotions.values()),
        },
        "sec_emotion": {
            "labels": list(sec_emotions.keys()),
            "values": list(sec_emotions.values()),
        },
        "genre": {
            "labels": list(genres_sorted.keys()),
            "values": list(genres_sorted.values()),
        },
        "speaker": {
            "labels": list(speakers.keys()),
            "values": list(speakers.values()),
        },
        "dur_hist": {"labels": dur_labels, "values": dur_counts},
        "wpm_hist": {"labels": wpm_labels, "values": wpm_counts},
        "lufs_hist": {"labels": delta_labels, "values": delta_counts},
        "snr_hist": {"labels": snr_labels, "values": snr_counts},
        "sc_hist": {"labels": sc_labels, "values": sc_counts},
    }

    return _HTML_TEMPLATE.replace("___CHART_DATA___", _json.dumps(chart_data))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TTS dataset")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all clips (default: approved only)")
    args = parser.parse_args()

    if not CLIPS_META_CSV.exists():
        sys.exit(f"ERROR: {CLIPS_META_CSV} not found. Run phases 1–5 first.")

    with open(CLIPS_META_CSV, newline="", encoding="utf-8") as f:
        all_clips = list(csv.DictReader(f))

    pool = all_clips if args.all else [c for c in all_clips if c.get("approved", "").lower() == "true"]

    if not pool:
        sys.exit("No clips to evaluate. Run the tagging app (05) to approve clips first.")

    print(f"Evaluating {len(pool)} {'clips (all)' if args.all else 'approved clips'}...")

    metrics = []
    for i, clip in enumerate(pool, 1):
        fn = clip.get("clip_filename", "")
        path = CLIPS_DIR / fn
        print(f"  [{i:3d}/{len(pool)}] {fn}", end="", flush=True)

        m = dict(clip)
        m.update(transcript_stats(clip.get("transcript", ""), float(clip.get("duration_s", 0) or 0)))

        if path.exists():
            aq = analyze_clip(path)
            if "error" in aq:
                print(f"  [ERR: {aq['error']}]", end="")
            else:
                m.update(aq)
                m["lufs_delta"] = round(m["lufs"] - (-23.0), 2) if not np.isnan(m["lufs"]) else float("nan")
            print(f"  LUFS={m.get('lufs', '?'):>7}  SNR={m.get('snr_db', '?'):>6} dB"
                  f"  SC={m.get('spectral_centroid_hz', '?'):>7} Hz"
                  f"  sil={m.get('silence_ratio', '?'):.3f}")
        else:
            print("  [FILE NOT FOUND]")

        metrics.append(m)

    # All clips (not just pool) needed for full markdown stats
    with open(CLIPS_META_CSV, newline="", encoding="utf-8") as f:
        all_meta = list(csv.DictReader(f))
    all_metrics = []
    pool_fns = {m["clip_filename"] for m in metrics}
    for c in all_meta:
        if c["clip_filename"] in pool_fns:
            all_metrics.append(next(m for m in metrics if m["clip_filename"] == c["clip_filename"]))
        else:
            m2 = dict(c)
            m2.update(transcript_stats(c.get("transcript", ""), float(c.get("duration_s", 0) or 0)))
            all_metrics.append(m2)

    repo_id = "nitya2405/sarvam-tts-dataset"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLIP_METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    print(f"\nPer-clip metrics: {CLIP_METRICS_CSV}")

    md = generate_markdown_report(all_metrics, include_all=args.all, repo_id=repo_id)
    REPORT_MD.write_text(md, encoding="utf-8")
    print(f"Markdown report:  {REPORT_MD}")

    html = generate_html_report(metrics, include_all=args.all, repo_id=repo_id)
    HTML_REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"HTML dashboard:   {HTML_REPORT_PATH}")


if __name__ == "__main__":
    main()

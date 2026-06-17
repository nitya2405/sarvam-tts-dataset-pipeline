"""
Phase 5: Gradio-based tagging app for reviewing clips, correcting transcripts, and assigning emotion tags.
Every action writes back to clips_metadata.csv immediately.

Usage:
    python scripts/05_tagging_app.py
    python scripts/05_tagging_app.py --port 7861
"""

import argparse
import csv
import os
from pathlib import Path

import gradio as gr
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
CLIPS_META_CSV = ROOT / "metadata" / "clips_metadata.csv"
CLIPS_DIR = ROOT / "data" / "clips"
SETTINGS = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())

EMOTION_TAGS = SETTINGS["emotion_tags"]

CLIPS_META_FIELDS = [
    "clip_filename", "language", "genre", "speaker_name",
    "source_index", "video_id", "source_channel", "source_url",
    "clip_start_time_s", "clip_end_time_s", "duration_s",
    "transcript", "primary_emotion", "secondary_emotion",
    "approved", "reviewed", "qc_flags",
]


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_meta() -> list[dict]:
    if not CLIPS_META_CSV.exists():
        return []
    with open(CLIPS_META_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_meta(rows: list[dict]) -> None:
    with open(CLIPS_META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLIPS_META_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# App state helpers
# ---------------------------------------------------------------------------

def find_unreviewed_index(rows: list[dict], current: int) -> int:
    """Return index of the next unreviewed clip after current (wraps to 0)."""
    n = len(rows)
    for offset in range(1, n + 1):
        idx = (current + offset) % n
        if rows[idx].get("reviewed", "false").lower() != "true":
            return idx
    return current  # all reviewed, stay put


def progress_text(rows: list[dict]) -> str:
    reviewed = sum(1 for r in rows if r.get("reviewed", "false").lower() == "true")
    approved = sum(1 for r in rows if r.get("approved", "").lower() == "true")
    return f"{reviewed}/{len(rows)} reviewed | {approved} approved"


def row_info(row: dict) -> str:
    qc = row.get("qc_flags", "")
    qc_str = f"\n**QC flags:** {qc}" if qc else ""
    return (
        f"**Clip:** {row.get('clip_filename', '')}\n"
        f"**Speaker:** {row.get('speaker_name', '')} | "
        f"**Lang:** {row.get('language', '')} | "
        f"**Genre:** {row.get('genre', '')}\n"
        f"**Duration:** {float(row.get('duration_s', 0)):.1f}s | "
        f"**Source:** {row.get('source_url', '')}"
        f"{qc_str}"
    )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    rows = load_meta()
    if not rows:
        raise RuntimeError(f"No clips found in {CLIPS_META_CSV}. Run phases 1-4 first.")

    state = {"rows": rows, "current": 0}

    # Jump to first unreviewed clip on startup
    for i, r in enumerate(rows):
        if r.get("reviewed", "false").lower() != "true":
            state["current"] = i
            break

    def get_clip_audio(idx: int) -> str | None:
        fn = state["rows"][idx].get("clip_filename", "")
        path = CLIPS_DIR / fn
        return str(path) if fn and path.exists() else None

    def load_clip(idx: int):
        rows = state["rows"]
        if not rows:
            return [None, "", "", None, None, "No clips", ""]
        idx = max(0, min(idx, len(rows) - 1))
        state["current"] = idx
        row = rows[idx]
        audio = get_clip_audio(idx)
        transcript = row.get("transcript", "")
        primary = row.get("primary_emotion", "")
        secondary = row.get("secondary_emotion", "")
        approved_val = row.get("approved", "")
        approved = True if approved_val.lower() == "true" else False
        info = row_info(row)
        prog = progress_text(rows)
        return audio, transcript, primary, secondary, approved, info, prog

    def save_and_advance(transcript, primary, secondary, approved, go_next):
        rows = state["rows"]
        idx = state["current"]
        row = rows[idx]

        row["transcript"] = transcript
        row["primary_emotion"] = primary or ""
        row["secondary_emotion"] = secondary or ""
        row["approved"] = "true" if approved else "false"
        row["reviewed"] = "true"

        save_meta(rows)

        if go_next:
            next_idx = find_unreviewed_index(rows, idx)
        else:
            next_idx = idx

        return load_clip(next_idx) + [next_idx]

    def nav_to(idx):
        idx = max(0, min(int(idx or 0), len(state["rows"]) - 1))
        return load_clip(idx) + [idx]

    def nav_prev():
        idx = max(0, state["current"] - 1)
        return load_clip(idx) + [idx]

    def nav_next_any():
        idx = min(len(state["rows"]) - 1, state["current"] + 1)
        return load_clip(idx) + [idx]

    def jump_next_unreviewed():
        idx = find_unreviewed_index(state["rows"], state["current"])
        return load_clip(idx) + [idx]

    with gr.Blocks(title="Sarvam TTS Dataset Tagger", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Sarvam TTS Dataset Tagger")

        with gr.Row():
            progress_label = gr.Markdown(progress_text(rows))

        with gr.Row():
            with gr.Column(scale=1):
                audio_player = gr.Audio(label="Clip Audio", type="filepath", autoplay=False)
                clip_info = gr.Markdown(row_info(rows[state["current"]]))

            with gr.Column(scale=1):
                transcript_box = gr.Textbox(
                    label="Transcript (edit to correct)",
                    lines=6,
                    value=rows[state["current"]].get("transcript", ""),
                )
                primary_tag = gr.Dropdown(
                    label="Primary Emotion Tag (required)",
                    choices=EMOTION_TAGS,
                    value=rows[state["current"]].get("primary_emotion") or None,
                )
                secondary_tag = gr.Dropdown(
                    label="Secondary Emotion Tag (optional)",
                    choices=[""] + EMOTION_TAGS,
                    value=rows[state["current"]].get("secondary_emotion") or "",
                )
                approved_check = gr.Checkbox(
                    label="Approve this clip",
                    value=rows[state["current"]].get("approved", "").lower() == "true",
                )

        with gr.Row():
            btn_prev = gr.Button("Prev", size="sm")
            current_idx_box = gr.Number(
                label="Jump to index", value=state["current"], precision=0, minimum=0,
                maximum=len(rows) - 1,
            )
            btn_next_any = gr.Button("Next", size="sm")
            btn_next_unreviewed = gr.Button("Next Unreviewed", variant="secondary")
            btn_save_advance = gr.Button("Save & Next Unreviewed", variant="primary")

        # Wire up outputs list (shared across all nav actions)
        outputs = [audio_player, transcript_box, primary_tag, secondary_tag,
                   approved_check, clip_info, progress_label, current_idx_box]

        btn_save_advance.click(
            fn=lambda t, p, s, a: save_and_advance(t, p, s, a, go_next=True),
            inputs=[transcript_box, primary_tag, secondary_tag, approved_check],
            outputs=outputs,
        )
        btn_next_unreviewed.click(
            fn=jump_next_unreviewed,
            inputs=[],
            outputs=outputs,
        )
        btn_prev.click(fn=nav_prev, inputs=[], outputs=outputs)
        btn_next_any.click(fn=nav_next_any, inputs=[], outputs=outputs)
        current_idx_box.submit(fn=nav_to, inputs=[current_idx_box], outputs=outputs)

        # Load first clip on app start
        app.load(
            fn=lambda: load_clip(state["current"]) + [state["current"]],
            inputs=[],
            outputs=outputs,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio tagging app for TTS dataset review")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    app = build_app()
    print(f"Starting tagging app at http://localhost:{args.port}")
    app.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()

# Sarvam TTS Dataset Pipeline

Builds a clean, tagged, single-speaker audio dataset (Indian English + Hindi) for TTS/ASR research.
Target: ~30 minutes each of Indian English and Hindi clips, 45–60 seconds each, emotion-tagged.

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies (requires ffmpeg on PATH)
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env            # then edit .env with your keys
```

### `.env` file

```
SARVAM_API_KEY=your_sarvam_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Never commit `.env`. It is gitignored.

---

## Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and on PATH (used by pydub for audio conversion)
  - Windows: `winget install Gyan.FFmpeg` or download from https://ffmpeg.org
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`
- **Sarvam AI API key** — get one at https://dashboard.sarvam.ai
- **HuggingFace token** — for Phase 6 push only

---

## Running the pipeline

Work through the phases in order. Each script is independently runnable.

### Phase 1 — Download audio

```bash
# Test on one source first:
python scripts/01_download.py --index 0

# Then download everything (--resume skips already-downloaded):
python scripts/01_download.py --resume
```

Output: `data/raw/` + `metadata/source_log.csv`

### Phase 2 — Diarize (speaker separation)

**Costs API credits. Always run `--dry-run` first.**

```bash
# Estimate cost and test on first file:
python scripts/02_diarize_segment.py --dry-run

# Diarize a single source:
python scripts/02_diarize_segment.py --index 0

# Diarize all (confirm cost first!):
python scripts/02_diarize_segment.py --all
```

Output: `data/segments/*_diarization.json`

**After Phase 2 — manual step required:**
Open each `_diarization.json` file and check the `speaker_summary` field.
It shows which `speaker_id` has the most speaking time.
Open `metadata/source_log.csv` and fill in the `target_speaker_label` column
for each row with the correct speaker ID (e.g. `SPEAKER_00`).

### Phase 3 — Preprocess and split clips

Requires `target_speaker_label` to be filled in `source_log.csv`.

```bash
# Process one source:
python scripts/03_preprocess_audio.py --index 0

# Process all:
python scripts/03_preprocess_audio.py --all
```

Output: `data/clips/*.wav` + `metadata/clips_metadata.csv`

Audio spec applied: 24 kHz mono 16-bit WAV, 80 Hz high-pass, −23 LUFS / −1 dBTP,
silence-trimmed with 175 ms padding, 45–60 s clips at natural pause boundaries.

### Phase 4 — Transcribe clips

**Costs API credits. Always run `--dry-run` first.**

```bash
python scripts/04_transcribe.py --dry-run

# Transcribe one clip:
python scripts/04_transcribe.py --clip en_news_000_0001.wav

# Transcribe all:
python scripts/04_transcribe.py --all
```

Clips are 45–60s so the REST API (< 30s limit) can't be used — both phases go through
the Batch API (async). Transcripts are written back to `metadata/clips_metadata.csv`
immediately after each clip completes.

### Phase 5 — Tag and approve clips (Gradio app)

```bash
python scripts/05_tagging_app.py
# Opens at http://localhost:7860
```

For each clip: play audio, correct the transcript, assign a primary emotion tag
(and optional secondary), then approve/reject. Progress is saved to `clips_metadata.csv`
after every action — safe to close and resume.

Emotion taxonomy: `neutral`, `formal`, `happy_excited`, `sad`, `angry`,
`calm_reverent`, `conversational_casual`, `intense_dramatic`

### Phase 6 — Build and push HuggingFace dataset

```bash
# Dry run (local only, no push):
python scripts/06_build_dataset.py --repo-id your-username/sarvam-tts-dataset --dry-run

# Push public dataset:
python scripts/06_build_dataset.py --repo-id your-username/sarvam-tts-dataset

# Push private dataset:
python scripts/06_build_dataset.py --repo-id your-username/sarvam-tts-dataset --private
```

Output: dataset pushed to Hub + `dataset_card/README.md` generated.

---

## Config files

| File | Purpose |
|---|---|
| `config/sources.csv` | Source YouTube videos with language, genre, speaker, emotion tags |
| `config/settings.yaml` | Audio specs, API settings, paths, emotion taxonomy |

---

## Data files (gitignored)

| Directory | Contents |
|---|---|
| `data/raw/` | Full downloaded audio (not committed) |
| `data/segments/` | Diarization JSON files (not committed) |
| `data/clips/` | Final processed WAV clips (pushed to HuggingFace, not git) |
| `metadata/source_log.csv` | Download log + target speaker labels |
| `metadata/clips_metadata.csv` | Per-clip metadata, transcripts, emotion tags, review status |

---

## Environment variables

| Variable | Required for | Description |
|---|---|---|
| `SARVAM_API_KEY` | Phases 2, 4 | Sarvam AI API subscription key |
| `HF_TOKEN` | Phase 6 | HuggingFace write token |

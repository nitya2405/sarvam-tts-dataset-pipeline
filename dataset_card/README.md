---
language:
- hi
- en
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- audio
- tts
- asr
- indian-english
- hindi
- emotion
pretty_name: Sarvam TTS Dataset
size_categories:
- 1K<n<10K
---

# Sarvam TTS Dataset

A curated dataset of clean, single-speaker audio clips in Indian English and Hindi,
assembled for training and evaluating text-to-speech (TTS) and speech recognition models.
Each clip is 45–60 seconds, loudness-normalized to broadcast standard, and tagged with a
primary emotion label by a human reviewer.

## Dataset Summary

| | |
|---|---|
| Total clips | 86 |
| Total duration | 74.4 minutes |
| Languages | Indian English (EN), Hindi (HI) |
| Created | 2026-06-18 |
| Audio format | 24 kHz mono 16-bit PCM WAV, −23 LUFS |

## Duration and Emotion by Language

**English — 47 clips, 40.6 minutes**

| Emotion | Clips |
|---|---|
| happy_excited | 12 |
| intense_dramatic | 7 |
| conversational_casual | 6 |
| sad | 7 |
| calm_reverent | 5 |
| neutral | 4 |
| formal | 3 |
| angry | 3 |

**Hindi — 39 clips, 33.8 minutes**

| Emotion | Clips |
|---|---|
| intense_dramatic | 8 |
| conversational_casual | 7 |
| happy_excited | 6 |
| calm_reverent | 5 |
| angry | 4 |
| sad | 4 |
| neutral | 3 |
| formal | 2 |

## Genre Distribution

Across 17 source videos spanning: speech (21), conversational (14), podcast (13),
news (11), dramatic (11), vlog (7), standup (6), cooking (2), poetry (1).

## Methodology

### 1. Source selection

Sources are publicly available YouTube videos featuring natural Indian English and Hindi
speech across diverse genres (news anchors, political speeches, stand-up comedy, poetry
recitals, cooking shows, vlogs, podcasts). Genre and speaker diversity was prioritised to
cover the full emotion taxonomy.

### 2. Download

Audio extracted with `yt-dlp` at the best available quality, then converted to 16 kHz
WAV for diarization input.

### 3. Speaker diarization and the single-speaker interpretation

**The central design decision in this pipeline is how to handle multi-speaker audio.**

YouTube recordings almost always contain more than one speaker — interviews, panel
discussions, call-in shows, and ambient audience noise are the norm. The brief asked for
"single-speaker audio clips for TTS," but this constraint is genuinely ambiguous when
applied to real-world sources: does it mean the *source video* must have one speaker, or
that each *output clip* must contain only one speaker's voice?

We resolved this as follows: **each output clip must contain exactly one speaker's
voice, but source videos may contain multiple speakers.** This allows a much wider pool
of sources and more natural speech, while still producing clean single-speaker clips
suitable for TTS.

Implementation:
- Speaker separation was performed with Sarvam AI's `saaras:v3` batch API
  (`mode=transcribe`, `with_diarization=True`), which returns a turn-by-turn transcript
  annotated with `speaker_id` labels.
- For each source, a human reviewer inspected the `speaker_summary` field of the
  diarization output (which reports total speaking time per speaker) and identified the
  **target speaker** — the single voice that should represent that source in the dataset.
  This was recorded in `source_log.csv` as `target_speaker_label`.
- Only turns attributed to the target speaker were retained. Turns from other speakers,
  including interviewers, audience, and background voices, were discarded.

This means every clip contains exactly one speaker's voice throughout. However, the
transcription reflects only what the target speaker said, which may include responses to
questions or reactions to other speakers present in the original recording.

### 4. Clip construction

After diarization:
- Adjacent turns from the target speaker separated by ≤ 3 s were merged into a single
  continuous segment.
- Long segments were split at natural silence boundaries (≥ 200 ms pause, threshold
  −40 dB) into clips of 45–60 s. The 45 s lower bound ensures enough context for
  prosody modelling; the 60 s upper bound fits within practical batch job limits and
  keeps annotation effort per clip tractable.

### 5. Audio processing

Each clip was processed in order:
1. Resample to 24 000 Hz mono 16-bit PCM WAV (the standard for TTS training).
2. High-pass filter at 80 Hz to remove low-frequency rumble and HVAC hum.
3. Loudness normalise to −23 LUFS integrated loudness (EBU R128), −1 dBTP true-peak
   ceiling — the same standard used in broadcast and podcast mastering.
4. Silence trim at clip edges with ~175 ms of padding preserved.

### 6. Quality control flags

Before writing each clip, the pipeline runs two automated checks and records the results
in the `qc_flags` column of the metadata CSV. Flagged clips are **not automatically
rejected** — the human reviewer made the final keep/discard decision in Phase 5.

| Flag | Condition | Meaning |
|---|---|---|
| `clipping` | Any sample ≥ 0.99 absolute value | The source recording hit full scale before normalisation; may have transient distortion |
| `low_snr` | Integrated loudness of the raw clip < −35 LUFS | Likely a poor-quality microphone or high ambient noise floor |

The presence of a QC flag in an approved clip means the reviewer listened and judged the
artefact acceptable (e.g., a single clipped transient in an otherwise clean recording).

### 7. Transcription

Verbatim ASR via Sarvam AI's `saaras:v3` batch API (`mode=verbatim`, diarization off).
Clips were grouped by language and batched into jobs of ≤ 20 files (the API upload
limit). The Sarvam SDK names output files after the uploaded filename, so
transcript–clip matching was done by filename lookup rather than by position to avoid
order-dependent misalignment across chunks.

### 8. Human tagging and approval

Each clip was reviewed in a purpose-built Gradio application. For every clip, the
reviewer:
- Listened to the audio and read the auto-generated transcript.
- Corrected any transcription errors (disfluencies, proper nouns, code-switching).
- Assigned a **primary emotion tag** from the taxonomy below (required to approve).
- Optionally assigned a **secondary emotion tag**.
- Approved or rejected the clip.

Only approved clips are included in this dataset. Clips with corrected transcripts
reflect the human-reviewed version, not the raw ASR output.

**Emotion taxonomy:** `neutral`, `formal`, `happy_excited`, `sad`, `angry`,
`calm_reverent`, `conversational_casual`, `intense_dramatic`.

## Data Fields

| Column | Type | Description |
|---|---|---|
| `audio` | Audio (24 kHz) | Audio clip |
| `clip_filename` | string | Original WAV filename |
| `language` | string | `en` or `hi` |
| `genre` | string | Source genre |
| `speaker_name` | string | Speaker name from source log |
| `source_index` | int | Index into source video list |
| `video_id` | string | YouTube video ID |
| `source_channel` | string | YouTube channel name |
| `source_url` | string | YouTube URL |
| `clip_start_time_s` | float | Start time in the original source audio (seconds) |
| `clip_end_time_s` | float | End time in the original source audio (seconds) |
| `duration_s` | float | Clip duration in seconds |
| `transcript` | string | Human-reviewed verbatim transcript |
| `primary_emotion` | string | Primary emotion tag |
| `secondary_emotion` | string | Secondary emotion tag (may be empty) |
| `approved` | bool | True = approved by human reviewer |
| `reviewed` | bool | True = clip was seen by reviewer |
| `qc_flags` | string | Pipe-delimited QC flags (`clipping`, `low_snr`), or empty |

## Audio Format

| Property | Value |
|---|---|
| Sample rate | 24 000 Hz |
| Channels | Mono |
| Bit depth | 16-bit PCM |
| Format | WAV (stored as parquet via HuggingFace `Audio` feature) |
| Loudness | −23 LUFS (EBU R128) |
| True peak | −1 dBTP |
| High-pass | 80 Hz |

## License and Copyright

**Audio clips**: Sourced from publicly available YouTube videos. The clips are provided
here for non-commercial research purposes under fair use. Copyright in the original audio
remains with the respective creators and YouTube channels. Downstream users who wish to
use the audio commercially should obtain clearance from the original creators.

**Metadata** (transcripts, emotion tags, QC flags, and all other structured data): This
is original annotation work released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: Nitya Shukla, 2026.

The CC-BY 4.0 tag on this repository applies to the metadata. We do not claim or grant
any rights over the audio content itself beyond what fair use permits.

## Pipeline

The full pipeline (download → diarize → preprocess → transcribe → tag → publish) is
open-source at [nitya2405/sarvam-tts-dataset-pipeline](https://github.com/nitya2405/sarvam-tts-dataset-pipeline).

## Citation

```
@dataset{sarvam-tts-dataset-2026,
  title  = {Sarvam TTS Dataset},
  author = {Shukla, Nitya},
  year   = 2026,
  url    = {https://huggingface.co/datasets/nitya2405/sarvam-tts-dataset},
  note   = {Indian English and Hindi single-speaker TTS clips, human-reviewed,
            emotion-tagged, 74.4 minutes total}
}
```

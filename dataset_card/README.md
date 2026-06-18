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

## Dataset Summary

- **Total clips**: 85
- **Total duration**: 73.5 minutes
- **Created**: 2026-06-18
- **Repository**: `nitya2405/sarvam-tts-dataset`

## Duration by Language

- **EN**: 39.6 minutes
- **HI**: 33.8 minutes

## Emotion Distribution

Each clip is tagged with one primary emotion from the taxonomy:
`neutral`, `formal`, `happy_excited`, `sad`, `angry`, `calm_reverent`,
`conversational_casual`, `intense_dramatic`.

- `happy_excited`: 18 clips
- `intense_dramatic`: 15 clips
- `conversational_casual`: 14 clips
- `calm_reverent`: 11 clips
- `sad`: 11 clips
- `angry`: 7 clips
- `formal`: 5 clips
- `neutral`: 4 clips

## Genre Distribution

- `speech`: 20 clips
- `conversational`: 14 clips
- `podcast`: 13 clips
- `news`: 11 clips
- `dramatic`: 11 clips
- `vlog`: 7 clips
- `standup`: 6 clips
- `cooking`: 2 clips
- `poetry`: 1 clips

## Methodology

1. **Sources**: Publicly available YouTube videos featuring Indian English and Hindi speakers
   across diverse genres (news, speeches, stand-up comedy, poetry, cooking, vlog, podcast).
2. **Download**: Audio extracted via yt-dlp at best available quality.
3. **Diarization**: Speaker separation using Sarvam's batch STT API (`saaras:v2.5`)
   with `with_diarization=True`. Target speakers manually identified per source.
4. **Segmentation**: Adjacent turns from the target speaker merged (gap < 1 s),
   then split into 45–60 s clips at natural pause/silence boundaries.
5. **Audio processing**:
   - Resampled to 24 000 Hz mono 16-bit PCM WAV
   - High-pass filter at 80 Hz (removes rumble/hum)
   - Loudness normalized to −23 LUFS, −1 dBTP true-peak ceiling
   - Silence trimmed at clip edges with ~175 ms padding
6. **Transcription**: Verbatim ASR via Sarvam API (`saarika:v2`), human-corrected
   during tagging.
7. **Tagging**: Human review of each clip using a Gradio tagging application.
   Clips approved/rejected individually; emotion tags assigned per the taxonomy above.

## Audio Format

| Property | Value |
|---|---|
| Sample rate | 24 000 Hz |
| Channels | Mono |
| Bit depth | 16-bit PCM |
| Format | WAV |
| Loudness | −23 LUFS |
| True peak | −1 dBTP |

## License & Copyright

Audio clips are sourced from publicly available YouTube videos under fair use
for non-commercial research purposes. Original copyrights remain with the respective
creators and channels. Dataset metadata (transcripts, tags) is released under CC-BY 4.0.

## Citation

If you use this dataset, please cite:

```
@dataset{sarvam-tts-dataset-2026,
  title  = {Sarvam TTS Dataset},
  year   = 2026,
  url    = {https://huggingface.co/datasets/nitya2405/sarvam-tts-dataset},
}
```

# Dataset Evaluation Report

Generated: 2026-06-27  
Repo: `nitya2405/sarvam-tts-dataset`  
Scope: **approved clips**

---

## 1. Collection Summary

| Metric | Value |
|---|---|
| Total clips in metadata | 291 |
| Reviewed | 291 (100%) |
| Approved | 72 (24%) |
| Clips evaluated (approved clips) | 72 |
| Total duration | 62.5 min |
| Unique speakers | 18 |
| Unique genres | 9 |
| Languages | en, hi |

### Duration stats (approved clips)

| Stat | Value |
|---|---|
| Min | 47.1 s |
| Max | 57.3 s |
| Mean | 52.1 s |
| Std | 1.7 s |

---

## 2. Language & Genre Distribution

| Language | Clips | Duration |
| -------- | ----- | -------- |
| EN       | 37    | 32.0 min |
| HI       | 35    | 30.5 min |

| Genre          | Clips | Share      |
| -------------- | ----- | ---------- |
| speech         | 20    | 20 (27.8%) |
| dramatic       | 12    | 12 (16.7%) |
| conversational | 11    | 11 (15.3%) |
| news           | 8     | 8 (11.1%)  |
| vlog           | 6     | 6 (8.3%)   |
| podcast        | 5     | 5 (6.9%)   |
| standup        | 5     | 5 (6.9%)   |
| poetry         | 3     | 3 (4.2%)   |
| cooking        | 2     | 2 (2.8%)   |

---

## 3. Speaker Distribution

| Speaker           | Clips |
| ----------------- | ----- |
| Anand Mahindra    | 7     |
| Vikas Sir         | 7     |
| Sunita Rawat      | 6     |
| Rohan Joshi       | 6     |
| Narendra Modi     | 6     |
| Ravish Kumar      | 6     |
| Dr Vivek Modi     | 5     |
| Kenny Sebastian   | 5     |
| Ravi Gupta        | 5     |
| Unknown Narrator  | 5     |
| Palki Sharma      | 3     |
| Arnab Goswami     | 2     |
| Kabita            | 2     |
| Harshita Garge    | 2     |
| Sudhanshu Trivedi | 2     |
| Nikhil Kamath     | 1     |
| Nayab Midha       | 1     |
| Shubham Shyam     | 1     |

---

## 4. Emotion Distribution

### Primary emotion

| Emotion               | Clips | Share      |
| --------------------- | ----- | ---------- |
| happy_excited         | 12    | 12 (16.7%) |
| conversational_casual | 11    | 11 (15.3%) |
| intense_dramatic      | 10    | 10 (13.9%) |
| calm_reverent         | 9     | 9 (12.5%)  |
| sad                   | 9     | 9 (12.5%)  |
| formal                | 8     | 8 (11.1%)  |
| neutral               | 7     | 7 (9.7%)   |
| angry                 | 6     | 6 (8.3%)   |

### Secondary emotion

| Emotion               | Clips | Share      |
| --------------------- | ----- | ---------- |
| neutral               | 19    | 19 (26.4%) |
| conversational_casual | 14    | 14 (19.4%) |
| intense_dramatic      | 14    | 14 (19.4%) |
| formal                | 8     | 8 (11.1%)  |
| sad                   | 5     | 5 (6.9%)   |
| calm_reverent         | 4     | 4 (5.6%)   |
| angry                 | 4     | 4 (5.6%)   |
| happy_excited         | 2     | 2 (2.8%)   |

### Primary emotion × language

| Emotion               | EN | HI |
| --------------------- | -- | -- |
| angry                 | 3  | 3  |
| calm_reverent         | 4  | 5  |
| conversational_casual | 6  | 5  |
| formal                | 4  | 4  |
| happy_excited         | 6  | 6  |
| intense_dramatic      | 6  | 4  |
| neutral               | 4  | 3  |
| sad                   | 4  | 5  |

---

## 5. Audio Quality Metrics

Measured: **72** clips

### Aggregate audio stats

| Metric                      | Mean   | Std    | Min    | Max    |
| --------------------------- | ------ | ------ | ------ | ------ |
| LUFS (target −23)           | -23.03 | 0.24   | -24.49 | -22.95 |
| True peak dBTP (ceiling −1) | -5.03  | 2.24   | -9.67  | -1.0   |
| RMS dBFS                    | -23.4  | 0.45   | -24.58 | -22.64 |
| Estimated SNR dB            | 42.33  | 12.47  | 18.8   | 70.6   |
| Spectral centroid Hz        | 1927.2 | 338.55 | 1057.4 | 2735.4 |
| Silence ratio               | 0.29   | 0.07   | 0.121  | 0.494  |

### Loudness conformance

| Check                    | Pass        | Fail     |
| ------------------------ | ----------- | -------- |
| LUFS within ±1 dB of −23 | 70 (97.2%)  | 2 (2.8%) |
| True peak ≤ −1 dBTP      | 72 (100.0%) | 0 (0.0%) |

### SNR quality bands

| Band          | Count      |
| ------------- | ---------- |
| Good (≥25 dB) | 70 (97.2%) |
| OK (15–25 dB) | 2 (2.8%)   |
| Poor (<15 dB) | 0 (0.0%)   |

---

## 6. Transcript / Speech Rate

| Stat | Word count | Words per min |
| ---- | ---------- | ------------- |
| Mean | 143.26     | 165.02        |
| Std  | 22.5       | 25.26         |
| Min  | 98         | 111.7         |
| Max  | 196        | 227.3         |

### Speech rate by genre

| Genre          | N  | Mean WPM | Std WPM |
| -------------- | -- | -------- | ------- |
| conversational | 11 | 143.45   | 31.53   |
| cooking        | 2  | 213.5    | 13.8    |
| dramatic       | 12 | 183.39   | 16.96   |
| news           | 8  | 168.89   | 10.6    |
| podcast        | 5  | 160.28   | 14.67   |
| poetry         | 3  | 167.47   | 14.62   |
| speech         | 20 | 154.35   | 19.16   |
| standup        | 5  | 188.46   | 18.09   |
| vlog           | 6  | 165.23   | 8.65    |

### Unusually fast speech (>180 WPM)

| Clip                           | WPM   | Genre          | Speaker          |
| ------------------------------ | ----- | -------------- | ---------------- |
| hi_cooking_013_0002.wav        | 227.3 | cooking        | Kabita           |
| en_dramatic_009_0006.wav       | 217.0 | dramatic       | Rohan Joshi      |
| hi_speech_014_0002.wav         | 208.0 | speech         | Vikas Sir        |
| hi_standup_015_0001.wav        | 203.4 | standup        | Ravi Gupta       |
| en_dramatic_009_0003.wav       | 201.6 | dramatic       | Rohan Joshi      |
| hi_standup_015_0011.wav        | 200.2 | standup        | Ravi Gupta       |
| hi_cooking_013_0001.wav        | 199.7 | cooking        | Kabita           |
| hi_dramatic_016_0002.wav       | 198.6 | dramatic       | Unknown Narrator |
| hi_standup_015_0006.wav        | 194.9 | standup        | Ravi Gupta       |
| en_conversational_004_0004.wav | 191.8 | conversational | Dr Vivek Modi    |
| hi_dramatic_018_0001.wav       | 191.5 | dramatic       | Shubham Shyam    |
| hi_standup_015_0004.wav        | 190.4 | standup        | Ravi Gupta       |
| hi_dramatic_016_0008.wav       | 188.4 | dramatic       | Unknown Narrator |
| en_poetry_017_0004.wav         | 188.1 | poetry         | Harshita Garge   |
| hi_news_011_0015.wav           | 185.4 | news           | Ravish Kumar     |
| en_conversational_004_0014.wav | 183.9 | conversational | Dr Vivek Modi    |
| hi_dramatic_016_0006.wav       | 183.3 | dramatic       | Unknown Narrator |
| en_podcast_006_0003.wav        | 180.7 | podcast        | Kenny Sebastian  |

---

## 7. Dataset Readiness Assessment

**No significant issues found.** The dataset looks balanced and clean.

---

_Report generated by `scripts/07_evaluate_dataset.py` on 2026-06-27._
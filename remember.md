# Audio + Speech-AI Glossary, TTS Dataset Primer

Notes to have open while building the Sarvam dataset. Organized bottom-up: raw audio first, then speech-processing concepts, then TTS-dataset-specific stuff, then Sarvam's actual API surface, then HuggingFace dataset terms.

---

## 1. Raw audio fundamentals

**Sample rate**
Audio is just a wave being measured (sampled) many times per second. Sample rate is how many measurements per second, in Hertz (Hz). Common values: 16,000 Hz (16kHz, standard for speech/ASR), 22,050 Hz, 44,100 Hz (CD quality, music). Higher isn't automatically better for speech, speech only needs frequencies up to about 8kHz to sound fully intelligible, so 16kHz or 22kHz is plenty and is what most TTS training pipelines expect. If your sources are 44.1kHz (most YouTube audio is), you'll downsample to whatever Sarvam's docs specify before sending to the ASR API, and you'll pick one consistent rate for your final dataset.

**Mono vs stereo**
Mono = one audio channel. Stereo = two channels (left/right), used to create spatial sound for music/film. Speech datasets should always be mono. If your YouTube source is stereo, you collapse it to mono (literally averaging or just taking one channel) before processing.

**Bit depth**
How much precision each sample has (16-bit is standard, 24-bit is higher precision, used in professional recording). For your purposes, 16-bit is the norm, you don't need to think about this much beyond making sure you're not accidentally working with a degraded/re-encoded file.

**File format / codec**
WAV is uncompressed, what you generally want for processing and final clips since there's no compression artifacts. MP3, OPUS, AAC are compressed formats (smaller file size, some quality loss). YouTube audio typically downloads as some compressed format (often OPUS or M4A), you'll want to convert to WAV for the actual processing/ASR steps, since compression can introduce subtle artifacts that hurt both ASR accuracy and downstream TTS training quality.

**Loudness / normalization**
If you took ten different YouTube clips, they'd likely all be at different volumes, some videos are mastered louder than others. Loudness normalization brings every clip to a consistent perceived volume. The standard unit for this is **LUFS** (Loudness Units Full Scale), a measurement of perceived loudness over time, not just peak volume. A typical target for spoken-word datasets is around -23 LUFS (broadcast standard) or -20 LUFS, the exact number matters less than every clip in your dataset landing at the *same* number.

**Clipping**
When a sound is too loud for the recording to capture, the waveform gets cut off (flattened) at the top and bottom instead of curving naturally. This sounds like distortion/crackling. Clipped audio is unusable for TTS training, you'd reject any clip with clipping, you can usually spot it by ear (a harsh, fuzzy quality on loud syllables) or by looking at the waveform (flat-topped peaks instead of rounded ones).

**Noise floor / SNR (Signal-to-Noise Ratio)**
"Signal" is the speech you want, "noise" is everything else (hiss, hum, background traffic, room echo, background music). SNR is a ratio describing how much louder the speech is compared to the background noise, expressed in dB. Higher SNR = cleaner audio. For TTS training data you want high SNR, ideally near-silent background. This is also why "no background music" is such a hard requirement, background music doesn't show up as random noise, it shows up as a second, very loud, very structured signal that the model can't separate from the voice.

**Silence trimming**
Most recordings have dead air at the start/end (someone clicking record, a pause before speaking). You trim this so each clip starts right as speech begins and ends right as it stops, with maybe a small buffer (100-300ms) so words don't feel clipped off.

---

## 2. Speech-processing concepts (the "AI" layer)

**VAD (Voice Activity Detection)**
A model/algorithm that listens to audio and outputs "speech happening here" vs "silence/non-speech here," basically a timestamp map of where someone is talking. Used to automatically find clean start/end points for segmenting a long recording into clips, instead of you manually scrubbing through a 20-minute video by hand.

**Diarization**
The process of figuring out *who* is speaking *when* in a recording with multiple speakers, it's not transcribing words, it's labeling stretches of audio as "Speaker A," "Speaker B," etc. This is the tool that lets you take a multi-speaker podcast or interview and pull out only the continuous stretches belonging to one speaker, which you then treat as your single-speaker source material.

**ASR (Automatic Speech Recognition)**
The actual speech-to-text step, audio in, written transcript out. This is what Sarvam's API does. Different ASR systems can output transcripts in different "styles" (see verbatim vs normalized below), which matters a lot for what you're building.

**Verbatim transcription vs normalized transcription**
This is an important distinction for your use case. A *normalized* transcript cleans things up for readability, removing filler words ("um," "uh"), fixing grammar, smoothing repetitions. A *verbatim* transcript captures exactly what was said, including fillers, false starts, repeated words, pauses. For TTS training data, you want verbatim, because the model needs the text to match the actual audio exactly, including the natural disfluencies that make speech sound human. If your transcript says "I went to the store" but the audio actually contains "I, uh, went to the- to the store," that mismatch is exactly the kind of error that degrades a TTS model trained on the pair.

**Transliteration vs Translation**
Translation changes the *language* (Hindi audio → English text, different words, same meaning). Transliteration keeps the *same language* but changes the *script* (Hindi audio → Hindi words, but written in Roman/Latin letters instead of Devanagari). You want neither for your dataset, you want transcription: same language, same script, matching exactly what's spoken.

**Code-switching / code-mixing**
When a speaker mixes two languages mid-sentence or mid-conversation, extremely common in Indian English and in everyday Hindi speech ("yaar this is so annoying," "main bas thoda busy hoon today"). Some ASR systems have a specific mode for this because it's a different problem than transcribing a single language cleanly, the system has to detect language switches on the fly. Worth knowing because your "Indian English" samples will very likely contain some Hindi (or other) words naturally, and you'll need to decide how to handle that in your transcripts and possibly your tagging.

**Speaker embedding**
A way of converting a chunk of someone's voice into a numerical fingerprint (a vector) that represents *how they sound*, independent of what words they're saying. Used to verify that two clips are actually the same speaker, useful as an automated sanity check after diarization, to make sure you didn't accidentally let a second speaker's voice slip into your "single speaker" clips.

---

## 3. TTS-dataset-specific concepts

**Why single-speaker matters**
A TTS model learns to map text to a voice. If your training clips contain switches between speakers, the model gets confused signals about what "the voice" sounds like. Every clip in a dataset like this must be confidently one person, start to finish.

**Why emotion/style tags matter**
A TTS model trained only on neutral, flat narration will only ever produce neutral, flat speech. If you want a model capable of expressive output (excited, sad, whispering, formal), it needs labeled examples of each style during training, so it learns to associate text patterns and speaker delivery with category labels you provide it.

**Segment length**
The "chunk size" of each audio sample in your dataset. Too short (a few seconds) and there's not enough context for prosody/emotion to read clearly. Too long (multiple minutes) and you risk a single clip drifting across multiple emotional tones, plus longer clips are harder to QC by ear and harder to align text-to-audio precisely. Most TTS datasets land somewhere in the 5-30 second range per clip, this assignment's own framing (60 one-minute samples, or 120 thirty-second samples) tells you they're flexible on this, the real constraint is *each clip should be one consistent emotional/stylistic delivery throughout*.

**Alignment**
Making sure the transcript text precisely corresponds to the audio clip's actual start and end, no extra words from before the clip started, no cut-off words at the boundary. Bad alignment is one of the most common silent failure modes in scraped datasets, a clip that starts half a syllable into a word looks fine on a spectrogram but sounds wrong and trains the model on a corrupted pairing.

**Prosody**
The "melody" of speech, pitch, rhythm, stress, intonation, the stuff that makes "are you serious?" sound like a genuine question vs sarcastic vs angry, even with identical words. This is essentially what your emotion/style tags are trying to capture in a labeled, structured way.

---

## 4. Sarvam API specifics (current, as of their docs today)

**Saaras v3**
Sarvam's current/recommended ASR model. Supports multiple output modes in one model: transcribe (plain transcription), translate, verbatim, transliterate, codemix. For your dataset, you'll mainly use **transcribe** or **verbatim** mode, verbatim is the one that preserves disfluencies and matches audio exactly, which is what you want for TTS training pairs.

**Saarika v2.5**
The older/legacy ASR model, being phased out. Don't build on this, use Saaras v3.

**REST API vs Batch API**
Sarvam offers two ways to call their speech-to-text:
- **REST API**: synchronous, instant response, but only works on audio under 30 seconds.
- **Batch API**: asynchronous (you submit a job, it processes, you poll or get a webhook callback when done), handles files up to 2 hours, and is the *only* mode that supports diarization.

Practically: you'll use the **Batch API with diarization on** for your raw, full-length downloaded YouTube audio (to split out single-speaker stretches), and then either Batch or REST (depending on your final clip length) for the clean, segmented clips you're about to add to your dataset.

**`with_diarization=True`**
The parameter you pass to get speaker-labeled output. The response includes speaker turns, basically a list of "Speaker X said Y between timestamp A and timestamp B," which is exactly the map you need to cut a multi-speaker recording into single-speaker pieces.

**Language codes**
Sarvam expects language codes like `hi-IN` (Hindi, India), and equivalents for other Indian languages. You'll set this per the language of the audio you're sending.

**Pricing/credits note**
Diarization is billed separately from plain transcription on the Batch API, worth keeping an eye on your free credits since you're processing potentially hours of raw source audio before you even get to your final 60 minutes of clips. If you're tight on credits, email Sarvam (the assignment explicitly says they'll top you up), better to ask early than run out mid-task.

---

## 5. HuggingFace dataset terms (for the publishing step)

**Dataset card**
The README.md that lives with your dataset on HuggingFace, describes what's in it, how it was built, licensing, intended use. This is also a place to demonstrate judgment, not just a formality.

**Audio feature type**
HF's `datasets` library has a built-in `Audio` column type that stores/loads audio files directly (rather than just file paths as plain strings), letting people stream and listen to samples directly from the dataset viewer in the browser. You want to use this properly so reviewers can actually listen without downloading anything.

**Splits**
Datasets are often divided into train/validation/test. For a dataset this size (60 minutes total), a full split setup is probably overkill, but it's worth deciding deliberately rather than leaving it undefined.

**Metadata columns**
Beyond audio + transcript, your dataset should carry extra columns: emotion/style tag, language, source video reference (ID/timestamp, not the raw link necessarily, but enough for traceability), duration. This is what turns "audio files in a folder" into an actual structured, usable dataset.

---

That's the full vocabulary you'll be working with. Once this all makes sense, we can move to actually building, starting with locking in your language and source genre list.
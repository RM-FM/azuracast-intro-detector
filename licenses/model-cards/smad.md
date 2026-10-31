---
library_name: transformers
pipeline_tag: audio-classification
tags:
- audio
- music
- speech
- singing-voice-detection
- speech-music-discrimination
- custom-code
license: mit
language:
- en
- vi
metrics:
- accuracy
- f1
model-index:
- name: smad-crnn
  results:
  - task:
      type: audio-classification
      name: Speech / music / singing discrimination
    dataset:
      type: custom
      name: SMAD synthetic test split (4,000 clips, source-disjoint)
    metrics:
    - type: accuracy
      value: 0.8795
      name: Accuracy
    - type: f1
      value: 0.8791
      name: Macro F1
---

# SMAD CRNN — speech / music / singing discrimination

A 834k-parameter CRNN that labels a 4-second audio clip as **spoken voice over
noise**, **spoken voice over music**, **sung voice over music**, or **no voice**.
Trained from scratch on log-mel spectrograms — no pretrained backbone, no
fine-tuning. Runs on CPU.

**▶ [Try it — upload a file, get a colour-coded timeline](https://huggingface.co/spaces/duclvQ/smad-demo)**

The point of the taxonomy: most speech/music segmenters merge "someone is
talking over a music bed" with "someone is singing", or call singing *music* and
stop there. SMAD separates those, which is what you need for lyric/dialogue
routing, dubbing QC, or MV vs. interview classification.

## Labels

| id | label | meaning |
|----|-------|---------|
| 0 | `speech_noise` | spoken voice over non-music background (noise, ambience, silence) |
| 1 | `speech_music` | spoken voice over a music bed |
| 2 | `singing_music` | sung voice (lyrics) over music |
| 3 | `none` | no human voice: instrumental music, noise, or silence |

Input: mono, 16 kHz, 4-second windows. Output: 4 logits; divide by
`config.temperature` (0.7095) before softmax for calibrated probabilities.

## Quick start

```python
import librosa, torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_id = "duclvQ/smad"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=True).eval()

audio, _ = librosa.load("clip.mp3", sr=16000, mono=True)   # mono float32 @ 16 kHz

inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits / model.config.temperature, dim=-1)[0]

label_id = int(probs.argmax())
print(model.config.id2label[label_id], float(probs[label_id]))
```

The feature extractor pads or truncates to exactly 4 seconds. For anything
longer, use the helper below.

## Audio of any length: sliding window + smoothing

Copy-paste. No dependencies beyond `librosa`, `numpy`, `torch`, `transformers`.

```python
import librosa, numpy as np, torch


def analyze(path, model, feature_extractor,
            hop_seconds=1.0, smooth_windows=5, batch_size=64):
    """Label a file of any length. Returns merged segments:
    [{"start": 0.0, "end": 12.0, "label": "singing_music", "confidence": 0.93}, ...]

    hop_seconds    step between windows (window itself is fixed at 4 s by training)
    smooth_windows median filter width over time; 1 disables smoothing
    """
    sr = feature_extractor.sampling_rate
    win = int(model.config.segment_seconds * sr)
    hop = max(int(hop_seconds * sr), 1)

    audio, _ = librosa.load(path, sr=sr, mono=True)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    starts = list(range(0, len(audio) - win + 1, hop))

    probs = []
    for i in range(0, len(starts), batch_size):
        batch = [audio[s:s + win] for s in starts[i:i + batch_size]]
        inputs = feature_extractor(batch, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs.append(torch.softmax(logits / model.config.temperature, dim=-1))
    probs = torch.cat(probs).numpy()

    # Median filter across time: kills isolated one-window flips, which are the
    # dominant error mode on real audio (a single drum fill read as `none`).
    if smooth_windows > 1:
        pad = smooth_windows // 2
        padded = np.pad(probs, ((pad, pad), (0, 0)), mode="edge")
        probs = np.stack([np.median(padded[i:i + smooth_windows], axis=0)
                          for i in range(len(probs))])

    # Merge runs of equal labels. A run ends where the next one starts, so the
    # returned segments tile the audio without overlapping -- consecutive
    # windows overlap by design whenever hop_seconds < 4.
    segments = []
    for start, p in zip(starts, probs):
        cid = int(p.argmax())
        if segments and segments[-1]["_id"] == cid:
            segments[-1]["_conf"].append(float(p[cid]))
        else:
            segments.append({"start": start / sr, "_id": cid, "_conf": [float(p[cid])]})

    duration = len(audio) / sr
    for cur, nxt in zip(segments, segments[1:]):
        cur["end"] = nxt["start"]
    segments[-1]["end"] = min(starts[-1] / sr + win / sr, duration)

    return [{"start": round(s["start"], 2), "end": round(s["end"], 2),
             "label": model.config.id2label[s["_id"]],
             "confidence": round(float(np.mean(s["_conf"])), 4)}
            for s in segments]


for seg in analyze("song.mp3", model, feature_extractor):
    print(f"{seg['start']:7.1f} - {seg['end']:7.1f}s  {seg['label']:<14s} {seg['confidence']:.2f}")
```

Segment boundaries are quantised to `hop_seconds`, and a boundary is only ever
accurate to within one 4-second window — that window is the model's unit of
decision, not a frame-accurate onset. Lower `hop_seconds` for finer boundaries
at linear cost; raise `smooth_windows` if the timeline flickers.

## Results

4,000 held-out clips, 1,000 per class, source-disjoint from training (no
speaker, song, or noise file appears in both). Chance is 25%.

**Overall accuracy: 87.95%** · macro F1 0.879 · 834,728 parameters

### Per class

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| `speech_noise` | 0.950 | 0.957 | **0.954** | 1000 |
| `speech_music` | 0.951 | 0.942 | **0.946** | 1000 |
| `none` | 0.782 | 0.867 | **0.823** | 1000 |
| `singing_music` | 0.841 | 0.752 | **0.794** | 1000 |

### Confusion matrix

Rows = true, columns = predicted.

| | `speech_noise` | `speech_music` | `singing_music` | `none` |
|---|---|---|---|---|
| **`speech_noise`** | **957** | 43 | 0 | 0 |
| **`speech_music`** | 47 | **942** | 11 | 0 |
| **`singing_music`** | 1 | 6 | **752** | 241 |
| **`none`** | 2 | 0 | 131 | **867** |

The two speech classes are close to solved. `singing_music` ↔ `none` holds 372
of the 482 total errors — 77%. Telling a sung voice apart from the instrumental
track under it is the open problem, not an implementation gap. If your use case
only needs "is anyone talking", the relevant number is the top-left 2×2 block.

### Accuracy by mixing difficulty

The test mixer records the SNR (or gain, for single-source clips) of each clip:

| bucket | accuracy | n |
|---|---|---|
| SNR [15,20] dB | 0.984 | 244 |
| SNR [5,10) dB | 0.962 | 547 |
| SNR [0,5) dB | 0.939 | 512 |
| SNR [10,15) dB | 0.938 | 448 |
| SNR [-5,0) dB | 0.932 | 249 |
| gain [0,5] dB (single-source) | 0.768 | 354 |
| gain [-5,0) dB (single-source) | 0.755 | 318 |
| gain [-10,-5) dB (single-source) | 0.732 | 328 |

Voice buried at negative SNR is *not* the hard case. Single-source clips are —
they are the `none` class, where there is no second source to key on.

### Calibration

Trained with label smoothing 0.1 + SpecAugment, then temperature-scaled on the
validation split (T = 0.7095, fitted — not tuned by hand).

| metric | raw | after temperature scaling |
|---|---|---|
| expected calibration error | 0.0404 | **0.0335** |
| errors made at >0.99 confidence | 0.0% | 1.0% |
| mean confidence when correct | — | 0.934 |
| mean confidence when wrong | — | 0.753 |

Why this matters: an earlier transformer trained *without* label smoothing put
>0.99 confidence on **24.5%** of its own mistakes — its confidence score was
useless as a filter, which is the first thing anyone actually wants from it.
This model's score is usable as a threshold. Temperature scaling is monotonic,
so it never changes which class wins.

## Compared with inaSpeechSegmenter

Same 1,000 clips, drawn from the test split above. The two systems don't share a
taxonomy — inaSpeechSegmenter tags singing as `music` **by design**, and merges
speech-over-music with speech-over-noise — so a naive head-to-head would be
meaningless. Two separate measurements instead:

**Task A — is a spoken voice present?** The question both systems are built to
answer. Neither is handicapped.

| system | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| inaSpeechSegmenter | 0.896 | 0.997 | 0.794 | 0.884 |
| **SMAD** | **0.996** | 0.998 | **0.994** | **0.996** |

**Task B — is *any* voice present, spoken or sung?** SMAD is built for this;
inaSpeechSegmenter is built *not* to distinguish it. Reporting B is not a
criticism of their tool — it quantifies what the extra class buys, which is the
only reason to carry it.

| system | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| inaSpeechSegmenter | 0.648 | 1.000 | 0.531 | 0.693 |
| **SMAD** | **0.908** | 0.957 | **0.919** | **0.937** |

On the 250 clips that genuinely contain singing, inaSpeechSegmenter labelled 228
`music`, 14 `noise`, 7 `noEnergy`, and **1** `speech`. SMAD got 188 right.

Size is comparable, not smaller: SMAD is 834,728 parameters (3.3 MB) against
inaSpeechSegmenter's speech/music/noise CNN at ~789k stored values (3.2 MB). The
gain here is the taxonomy and the recall, not the footprint.

*Caveat, stated plainly:* the clips come from SMAD's own synthetic mixer, so this
is home turf. inaSpeechSegmenter was trained on different data for a different
label set and never saw this distribution. Read this as "what the extra class
buys on this task", not as a general ranking.

## Training data

40,000 synthetic 4-second mixtures. The classes are defined by *what is mixed
together*, so segments are synthesised from clean stems rather than scraped from
labelled corpora — that gives frame-exact labels and direct control over mixing
SNR. Splits are disjoint at the **source** level (speaker id, song, noise file),
not the clip level.

| role | corpus | license |
|---|---|---|
| speech (EN) | [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr) `train.clean.100` | CC BY 4.0 |
| speech (VI) | Common Voice VI | CC0 1.0 |
| speech (VI) | FLEURS VI | CC BY 4.0 |
| singing + accompaniment | [`danjacobellis/musdb18HQ`](https://huggingface.co/datasets/danjacobellis/musdb18HQ) | **CC BY-NC-SA 4.0 — non-commercial** |
| instrumental music | [`benjamin-paine/free-music-archive-small`](https://huggingface.co/datasets/benjamin-paine/free-music-archive-small), `instrumental == Yes` | per-track Creative Commons; metadata MIT |
| noise | [`FluidInference/musan`](https://huggingface.co/datasets/FluidInference/musan) (noise subset only) | CC BY 4.0 |
| music (VI) | Wikimedia Commons Vietnamese traditional music | CC BY-SA / CC0, per file |
| real-world | crawled MV / news audio, pseudo-labelled with Silero VAD, **training split only** | no license grant — see below |

**Before you deploy this commercially, read this.** Two ingredients constrain it:

1. **MUSDB18-HQ is CC BY-NC-SA 4.0 — non-commercial research only.** It is the
   only separated-stem source in the pipeline and supplies both halves of
   `singing_music`. Whether a model trained on NC-licensed audio is itself
   encumbered is unsettled and jurisdiction-dependent. Get your own advice.
2. **The crawled real-world subset carries no license grant.** It went into
   *training only* — validation and test stay purely synthetic, so every number
   above remains comparable with every number measured before and after it was
   added. Each crawled file has a JSON sidecar recording source URL, video id,
   uploader, license field, the search query that surfaced it, and a sha256, so
   any source can be audited and removed on request.

The `license: mit` tag on this repo covers the **code and weights as published
by the author**; it does not and cannot re-license the upstream training corpora.

## Limitations

- **87.95% is an upper bound, not an estimate.** Train and test come from the
  same synthetic mixer and share its biases: no room impulse responses, one SNR
  distribution, one pool of speakers and songs. Expect real-world numbers below
  this. Evaluation on genuine recordings (e.g. MIR-1K) is the honest next step
  and has not been done.
- `singing_music` is capped by data, not architecture: only 281 distinct sources
  exist for that class project-wide, because MUSDB18-HQ has 150 tracks.
- Speech coverage is English + Vietnamese. Other languages are untested.
- The 4-second window is fixed by training. Shorter clips are zero-padded;
  a 4-second window straddling a speech→music cut gets one label for both.
- Music with heavy vocal-like synths, and speech over strongly rhythmic
  ambience, are the known confusions beyond the `singing_music`/`none` pair.

## Architecture

`TinyAudioCRNN`: 4 conv blocks (32→64→128→128, BN + ReLU + max-pool) over an
80-bin log-mel spectrogram, then a 1-layer BiGRU (hidden 128), then mean+max
pooling over time into a linear classifier. Per-mel-bin input standardisation
(`feat_mean` / `feat_std`) ships **inside** the weights rather than in a config
file, so inference physically cannot feed the model a different input scale than
it trained on.

Features must match training exactly: 80 mels, `n_fft=400` (25 ms),
`hop_length=160` (10 ms), `power=2.0`, `librosa.power_to_db`. The bundled feature
extractor does this for you.

Ablations at the same training recipe: transformer (683k params) 82.53%;
transformer without label smoothing + SpecAugment 80.13%. So the recipe is worth
+2.40 points and the CRNN architecture +5.42.

Selected at epoch 15 of 24 by validation accuracy (90.48%); early-stopped at 23.
~10 minutes on an RTX A4500.

## Citation

```bibtex
@software{smad_crnn,
  title  = {SMAD: a small CRNN for speech / music / singing discrimination},
  author = {duclvQ},
  year   = {2026},
  url    = {https://huggingface.co/duclvQ/smad}
}
```

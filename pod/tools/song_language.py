"""One spoken/sung language per recording, detected on its first-30s vocal stem.

All Whisper and Qwen passes of a recording use this language, so the four ASR
inputs stay comparable. Detection follows openai-whisper's own
``transcribe(language=None)`` path on the cached Mel-Band stem; no extra
separation or transcription pass. The 0-30s analysis window is unchanged.
"""
import hashlib
import json
from pathlib import Path
import re
import time

import whisper_experiment as baseline

PROFILE = 'whisper-turbo-lid-vocal-stem-30s-v1'
PER_SONG = 'song'
# Retry language when the song language yields no usable words, e.g. English lines of a
# mixed-language song; it is also the former fixed language, so its caches are reused.
FALLBACK = 'en'
# Subtitle/caption credits that Whisper invents in quiet audio or a mismatched language.
CREDITS = re.compile(
    r"sous-?titr|untertitel|subt[ií]tul|sottotitol|legendas? (?:pela|por)|ondertitel|napisy|"
    r"amara\.org|subtitles? by|ご視聴|시청해\s*주셔서|字幕", re.I)


def needs_fallback(language, word_count, text):
    """No lexical words, or nothing but an invented credit, in a non-fallback language."""
    if not language or language == FALLBACK:
        return False
    text = str(text or '')
    match = CREDITS.search(text)
    before = text[:match.start()] if match else text
    return word_count == 0 or not any(c.isalnum() for c in before)


def with_fallback(attempt, language, per_song):
    """attempt(language) -> (identity, path, stored, reused, summary).

    Returns (language used, original language if it fell back, attempt, seconds this run).
    """
    first = attempt(language)
    seconds = lambda *runs: sum(0 if run[3] else run[2]['elapsed_seconds'] for run in runs)  # noqa: E731
    if not (per_song and needs_fallback(language, first[4]['word_count'], first[2]['result'].get('text'))):
        return language, None, first, seconds(first)
    second = attempt(FALLBACK)
    return FALLBACK, language, second, seconds(first, second)

# Whisper language codes for which Qwen3-ASR accepts a forced language.
QWEN_NAMES = {
    'zh': 'Chinese', 'en': 'English', 'yue': 'Cantonese', 'ar': 'Arabic', 'de': 'German',
    'fr': 'French', 'es': 'Spanish', 'pt': 'Portuguese', 'id': 'Indonesian', 'it': 'Italian',
    'ko': 'Korean', 'ru': 'Russian', 'th': 'Thai', 'vi': 'Vietnamese', 'ja': 'Japanese',
    'tr': 'Turkish', 'hi': 'Hindi', 'ms': 'Malay', 'nl': 'Dutch', 'sv': 'Swedish',
    'da': 'Danish', 'fi': 'Finnish', 'pl': 'Polish', 'cs': 'Czech', 'tl': 'Filipino',
    'fa': 'Persian', 'el': 'Greek', 'ro': 'Romanian', 'hu': 'Hungarian', 'mk': 'Macedonian',
}


def qwen_language(code):
    """Qwen name for a Whisper code; None lets Qwen detect an unsupported language itself."""
    return QWEN_NAMES.get(code) if code else None


def aligner_language(language):
    """Language passed to the Qwen forced aligner, which uses it only to split words.

    Korean would import soynlp (GPL-3.0). Whitespace splitting, used for every
    other space-delimited language, also separates Korean words and only the
    first word's start is evaluated here.
    """
    if language and 'korean' in language.casefold():
        return 'English'
    return language


def identity_for(audio_hash, duration, versions, source_input):
    return dict(profile=PROFILE, audio_sha256=audio_hash, duration=duration, input=source_input,
                model='turbo', packages=versions,
                source_sha256=baseline.sha256(Path(__file__)),
                decoder_sha256=baseline.sha256(Path(__file__).with_name('model_worker.py')))


def detect(model, samples):
    """Whisper's transcribe() language decision: first 30s mel segment, FP16 on CUDA."""
    import torch
    import whisper
    from whisper.audio import N_FRAMES, N_SAMPLES
    mel = whisper.log_mel_spectrogram(samples, model.dims.n_mels, padding=N_SAMPLES)
    segment = whisper.pad_or_trim(mel, N_FRAMES).to(model.device).to(torch.float16)
    _, probabilities = model.detect_language(segment)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return dict(language=ranked[0][0], probability=float(ranked[0][1]),
                top=[dict(language=code, probability=float(p)) for code, p in ranked[:5]])


def song_language(root, audio, audio_hash, duration, versions, source_input, load_model):
    """Cached per-recording language; load_model() is called only on a cache miss."""
    from model_worker import decode
    identity = identity_for(audio_hash, duration, versions, source_input)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = root / 'languages' / f'{key}.json'
    stored = baseline.cached_json(path, identity)
    if stored is None:
        model = load_model()
        began = time.monotonic()
        result = detect(model, decode(Path(audio), 16000, duration=duration))
        stored = dict(status='complete', identity=identity, result=result,
                      elapsed_seconds=time.monotonic() - began)
        path.parent.mkdir(exist_ok=True)
        baseline.atomic_json(path, stored)
        return dict(stored['result'], input=source_input, cached=False,
                    evidence=str(path.relative_to(root)))
    return dict(stored['result'], input=source_input, cached=True, evidence=str(path.relative_to(root)))

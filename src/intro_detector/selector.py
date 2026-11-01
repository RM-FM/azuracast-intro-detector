"""First-vocal selector: a frozen ExtraTrees ranking model plus lead-vocal and early-vocal rules."""
import gzip
from importlib import resources
import json
import math
import re

from .config import VocalCalibration
from .core import selector_model as core
from .evidence import INPUTS, compact

TRAINING_POLICY = core.POLICY
POLICY = f'{TRAINING_POLICY}+lead-vocal-guard-v3'
TRAINING_SCHEMA = 1
BAD_TIMING_FLAGS = {'invalid_word_timing', 'nonmonotonic_word_timing',
                    'text_without_alignment', 'alignment_failed'}
DEFAULT_VOCALS = VocalCalibration()


def default_training():
    return resources.files('intro_detector').joinpath('data/selector-training-v1.json.gz')


class Selector:
    def __init__(self, training=None, vocals=DEFAULT_VOCALS):
        import numpy as np
        self.vocals = vocals
        source = training or default_training()
        with source.open('rb') as raw, gzip.GzipFile(fileobj=raw) as stream:
            data = json.loads(stream.read())
        if data.get('schema') != TRAINING_SCHEMA or data.get('policy') != TRAINING_POLICY:
            raise ValueError('Selector training data does not match the selector model')
        groups = [dict(times=np.array(g['times']), features=np.array(g['features']), label=g['label'])
                  for g in data['groups']]
        self.training_songs = len(groups)
        self.model = core.fit(groups)
        self._np = np

    def predict(self, report_row):
        return self.predict_compact(compact(report_row), report_row)

    def predict_compact(self, row, report_row=None):
        times, features = core.group(row)
        times, features = list(times), list(features)
        early_vocals = _early_vocal_support(row, report_row) if report_row is not None else True
        if not early_vocals and any(cue >= 1.0 for cue in times):
            keep = [index for index, cue in enumerate(times) if cue >= 1.0]
            times = [times[index] for index in keep]
            features = [features[index] for index in keep]
        lead_events = _lead_word_events(report_row, self.vocals.adlib_words) if report_row is not None else []
        lead_events = [event for event in lead_events if early_vocals or event[0] >= 1.0]
        for cue, _ in lead_events:
            if not any(abs(cue - existing) < .01 for existing in times):
                times.append(cue)
                features.append(core.features(row, cue))
        scores = self.model.predict(features)
        index = int(self._np.argmax(scores))
        cue = float(times[index])
        lead_anchor, support = _lead_vocal_anchor(row, lead_events, report_row)
        guarded = (lead_anchor is not None
                   and not lead_anchor - 1.0 <= cue <= lead_anchor)
        if guarded:
            eligible = [i for i, candidate in enumerate(times)
                        if lead_anchor - 1.0 <= candidate <= lead_anchor]
            if eligible:
                index = max(eligible, key=lambda i: scores[i])
                cue = float(times[index])
        conflict = core.early_conflict(row, cue)
        if conflict and report_row is not None and _early_panns_are_unverified(row, report_row, cue):
            conflict = False
        early_vocal_s = None
        if conflict and report_row is not None:
            bursts = _vocal_bursts(row, cue - .5, self.vocals)
            if bursts:
                conflict = False
                early_vocal_s = _early_vocal_onset(bursts, self.vocals)
                cue = cue if early_vocal_s is None else early_vocal_s
        conflict = conflict or (report_row is not None and cue < 1.0 and not early_vocals)
        return dict(cue_s=cue, decision_s=None if conflict else cue, conflict=conflict,
                    rank_score=round(float(scores[index]), 4), provenance='development_fit', policy=POLICY,
                    lead_anchor_s=lead_anchor, lead_anchor_support=support,
                    lead_guard_applied=guarded and bool(eligible), early_vocal_s=early_vocal_s)


def _lead_word_events(report_row, adlib_words=DEFAULT_VOCALS.adlib_words):
    if not report_row:
        return []
    events = []
    for source, result in report_row.get('asr', {}).items():
        if source.split('/', 1)[0] not in INPUTS or _unreliable_lead_asr(result):
            continue
        words = result.get('words') or []
        if not words:
            continue
        tokens = [re.findall(r'[^\W_]+', str(word.get('word') or '').casefold()) for word in words]
        prefix = 0
        while prefix < len(tokens) and tokens[prefix] and all(
                token in adlib_words for token in tokens[prefix]):
            prefix += 1
        if prefix < 3 or prefix == len(words):
            prefix = 0
        word = words[prefix]
        start = word.get('start_original', word.get('start'))
        if (isinstance(start, (int, float)) and not isinstance(start, bool)
                and math.isfinite(start) and 0 <= start < report_row.get('duration', 0)):
            events.append((float(start), source.split('/', 1)[0]))
    return events


def _unreliable_lead_asr(result):
    flags = set(result.get('quality_flags') or [])
    if BAD_TIMING_FLAGS.intersection(flags) or 'compression_ratio' in flags:
        return True
    text = ' '.join(re.findall(r'[^\W_]+', str(result.get('text') or '').casefold()))
    if any(phrase in text for phrase in core.bad):
        return True
    tokens = re.findall(r'[^\W_]+', text)
    if (len(tokens) <= 2 and tokens and len(set(tokens)) == 1
            and len(tokens[0]) == 1 and tokens[0] not in {'a', 'i'}):
        return True
    return len(tokens) >= 8 and max((tokens.count(token) for token in set(tokens)), default=0) / len(tokens) >= .8


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _lyrics_from_start(result):
    """Reliable words flowing from the first second; a lone 0 s word before a long gap is misalignment."""
    if _unreliable_lead_asr(result):
        return False
    starts = [word.get('start_original', word.get('start')) for word in result.get('words') or []]
    starts = [start for start in starts if _number(start)]
    return len(starts) >= 3 and 0 <= starts[0] < 1 and starts[2] <= 4


def _vocal_bursts(row, end, vocals=DEFAULT_VOCALS):
    """(start, end, peak) spans before `end` where at least two inputs show vocal activity."""
    curves = []
    for variant in core.variants:
        acoustic = row.get('acoustic', {}).get(f'{variant}/4') or {}
        step = acoustic.get('frame_seconds')
        curve = (acoustic.get('curves') or {}).get('vocal') or []
        if _number(step) and step > 0 and curve:
            curves.append((step, curve))
    if len(curves) < 2:
        return []
    grid = min(step for step, _ in curves)
    bursts, start, peak = [], None, 0.
    for frame in range(max(0, int(end / grid)) + 1):
        time = round(frame * grid, 4)
        indexes = [(int(time / step + 1e-6), curve) for step, curve in curves]
        values = [curve[i] for i, curve in indexes if i < len(curve) and _number(curve[i])]
        if time < end and sum(value >= vocals.vocal_level for value in values) >= 2:
            start = time if start is None else start
            peak = max([peak, *values])
        elif start is not None:
            bursts.append((start, time, peak))
            start, peak = None, 0.
    return [burst for burst in bursts if burst[1] - burst[0] >= vocals.min_burst_seconds]


def _early_vocal_onset(bursts, vocals=DEFAULT_VOCALS):
    """A single short weak burst is an ad-lib to ignore; repeated or strong bursts start the vocals."""
    strong = any(end - start >= vocals.strong_burst_seconds or peak >= vocals.strong_burst_level
                 for start, end, peak in bursts)
    return bursts[0][0] if len(bursts) >= 2 or strong else None


def _early_vocal_support(row, report_row):
    if report_row is None:
        return True
    acoustic_variants = set()
    for variant in core.variants:
        acoustic = row.get('acoustic', {}).get(f'{variant}/4') or {}
        step = acoustic.get('frame_seconds')
        curve = (acoustic.get('curves') or {}).get('vocal') or []
        if not isinstance(step, (int, float)) or step <= 0 or not curve:
            continue
        end = min(len(curve), max(1, int(1.0 / step) + 1))
        if max(curve[:end], default=0.) >= .1:
            acoustic_variants.add(variant)
    if len(acoustic_variants) >= 2:
        return True

    aligned_variants = set()
    for source, result in report_row.get('asr', {}).items():
        if source.split('/', 1)[0] not in INPUTS:
            continue
        start = result.get('first_word_start')
        if (_unreliable_lead_asr(result) or not isinstance(start, (int, float))
                or isinstance(start, bool) or not math.isfinite(start) or not 0 <= start < 1):
            continue
        aligned_variants.add(source.split('/', 1)[0])
    lyric_variants = {source.split('/', 1)[0] for source, result in report_row.get('asr', {}).items()
                      if source.split('/', 1)[0] in INPUTS and _lyrics_from_start(result)}
    return ((len(acoustic_variants) >= 1 and len(aligned_variants) >= 2)
            or len(lyric_variants) >= 3
            or (len(lyric_variants) >= 2 and _smad_singing_support(report_row, 0.0)))


def _smad_singing_support(report_row, cue):
    smad = (report_row or {}).get('smad') or {}
    order = smad.get('class_order') or []
    try:
        singing_index = order.index('singing_music')
    except ValueError:
        return False
    window_seconds = smad.get('window_seconds', 4.0)
    return any(start <= cue + .25 and start + window_seconds >= cue - .25
               and len(probability) > singing_index and probability[singing_index] >= .5
               for start, probability in zip(smad.get('window_starts') or [],
                                            smad.get('probabilities') or []))


def _lead_vocal_anchor(row, events, report_row=None):
    if not events:
        return None, 0
    smad = row.get('smad') or {}
    order = smad.get('class_order') or []
    starts, probabilities = smad.get('window_starts') or [], smad.get('probabilities') or []
    window_seconds = smad.get('window_seconds', 4.0)
    try:
        singing_index = order.index('singing_music')
    except ValueError:
        singing_index = None

    for cue, _ in sorted(events):
        cluster = [(time, variant) for time, variant in events if abs(time - cue) <= .75]
        variants = {variant for _, variant in cluster}
        singing_support = singing_index is not None and any(
            start <= cue + .25 and start + window_seconds >= cue - .25
            and len(probability) > singing_index and probability[singing_index] >= .5
            for start, probability in zip(starts, probabilities))
        early_lyric_support = (cue < 1.0 and len(_early_lyric_variants(report_row)) >= 2
                               and singing_support)
        acoustic_support = False
        for variant in variants:
            for context in (4, 8):
                acoustic = row.get('acoustic', {}).get(f'{variant}/{context}') or {}
                step = acoustic.get('frame_seconds')
                curve = (acoustic.get('curves') or {}).get('vocal') or []
                if not isinstance(step, (int, float)) or step <= 0 or not curve:
                    continue
                first = max(0, int((cue - .25) / step))
                last = min(len(curve), int((cue + .75) / step) + 1)
                if last > first and max(curve[first:last]) >= .1:
                    acoustic_support = True
                    break
            if acoustic_support:
                break
        corroborated = ((acoustic_support and len(variants) >= 2)
                        or (singing_support and len(variants) >= 3) or early_lyric_support)
        if corroborated:
            return min(time for time, _ in cluster), len(variants)
    return None, 0


def _early_lyric_variants(report_row):
    return {source.split('/', 1)[0] for source, result in (report_row or {}).get('asr', {}).items()
            if source.split('/', 1)[0] in INPUTS and _lyrics_from_start(result)}


def _early_panns_are_unverified(row, report_row, cue):
    candidates = {item.get('chain'): item.get('cue_s') for item in report_row.get('candidates', [])}
    early = [candidates.get(f'{variant}/panns-4s/0.1/level') for variant in core.variants]
    early = [point for point in early if point is not None and point < cue - 8]
    return bool(early) and all(0 <= point <= 1 for point in early) and not _early_vocal_support(row, report_row)

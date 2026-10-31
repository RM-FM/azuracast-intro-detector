"""Turn a selector decision plus stored evidence into an intro marker."""
import math
import re
from ..evidence import INPUTS

POLICY = 'pod-vocal-musical-three-inputs-30s-v3'
# Vocals starting earlier leave no usable intro.
NO_INTRO_BELOW_S = 2.


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def clear_opening(evidence):
    """Require independent acoustic/singing checks, not an empty transcript."""
    if evidence.get('duration', 0) < 29.9:
        return False
    asr = evidence.get('asr', {})
    if not asr:
        return False
    hallucinations = {'thank you', 'to be continued', 'we ll be right back', 'we will be right back'}
    for name, row in asr.items():
        if name.split('/', 1)[0] not in INPUTS:
            continue
        text = ' '.join(re.findall(r'\w+', str(row.get('text', '')).casefold()))
        if text and text not in hallucinations:
            return False
        if row.get('first_word_start') is not None and not text:
            return False
    acoustic = evidence.get('acoustic', {})
    for name in ('original/4', 'demucs/4', 'melband/4'):
        row = acoustic.get(name, {})
        scores = row.get('curves', {}).get('vocal', [])
        step = row.get('frame_seconds', 0)
        if not scores or not number(step) or step <= 0 or len(scores)*step < 29.9:
            return False
        if any(not number(x) or x < 0 or x > 1 for x in scores):
            return False
        # No sustained vocal activity, even when ASR produced a stock hallucination.
        if max(scores) > .2 or sum(x > .1 for x in scores)*step > .15:
            return False
    smad = evidence.get('smad') or {}
    order = smad.get('class_order', [])
    if 'none' not in order or 'singing_music' not in order:
        return False
    probs, starts = smad.get('probabilities', []), smad.get('window_starts', [])
    if not probs or len(starts) != len(probs) or starts[0] > .1 or starts[-1]+smad.get('window_seconds', 4) < 29.9:
        return False
    return all(len(p) == len(order) and all(number(x) and 0 <= x <= 1 for x in p)
               and p[order.index('none')] >= .9 and p[order.index('singing_music')] <= .1 for p in probs)


def resolve(raw, evidence=None, music=None, reviewed=False, instrumental_review=False,
            no_intro_below=NO_INTRO_BELOW_S, music_window=(3., 27.)):
    result = dict(intro_origin='pod', intro_policy=POLICY, intro_type=None,
                  intro_kind='exact', intro_end=None, intro_status='needs_review',
                  intro_reason='missing_pod_result', usable=False, reviewed=False)
    if not raw:
        return result
    result['intro_candidate'] = raw.get('cue_s')
    if raw.get('conflict') and not instrumental_review:
        result['intro_reason'] = 'vocal_conflict_veto'
        return result
    if instrumental_review:
        # A listener can correct false vocals, but cannot invent a measured cue.
        duration = (music or {}).get('duration')
        if not number(duration) or duration < 29.9:
            result['intro_reason'] = 'reviewed_instrumental_missing_musical_measurement'
            return result
        result.update(intro_clearance_source='listener_instrumental',
                      intro_instrumental_review_policy='sha256-first30-v1',
                      intro_instrumental_reviewed=True,
                      intro_vocal_veto_overridden=bool(raw.get('conflict')))
    clear = instrumental_review or reviewed or clear_opening(evidence or {})
    if clear:
        candidate = (music or {}).get('candidate') or {}
        t = candidate.get('time_s')
        if number(t) and music_window[0] <= t <= music_window[1]:
            result.update(intro_type='musical', intro_end=t, intro_status='confirmed',
                          intro_reason=candidate.get('kind', 'musical_handover'), usable=True)
        else:
            result.update(intro_type='30s+', intro_kind='at_least', intro_at_least_s=30.,
                          intro_end=30., intro_status='confirmed', intro_reason='verified_clear_opening', usable=True)
        result.update(intro_clearance_verified=True, clear_until_s=30.,
                      reviewed=reviewed or instrumental_review)
        return result
    t = raw.get('decision_s')
    if number(t) and 0 <= t <= 30:
        usable = t >= no_intro_below
        result.update(intro_type='vocal', intro_end=t if usable else None,
                      intro_status='confirmed' if usable else 'not_detected',
                      intro_reason='pod_vocal' if usable else 'no_intro', usable=usable)
        if not usable:
            result['intro_candidate'] = t
    else:
        result['intro_reason'] = 'unresolved_pod_evidence'
    return result

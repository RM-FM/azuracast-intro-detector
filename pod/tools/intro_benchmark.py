#!/usr/bin/env python3
"""CPU-only candidate comparison, listening annotations and scorecards. No cue writes."""
import argparse
import csv
import datetime
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import statistics

import whisper_experiment as base
import stem_comparison as stems
import qwen_experiment as qwen
from ultimate_inference import INPUTS, read, root_for

POLICY = 'intro-benchmark-v2-smad-auto-language'
LABEL_FIELDS = ('first_vocal_s', 'first_word_s', 'radio_handover_s')
BAD_TIMING = {'invalid_word_timing', 'nonmonotonic_word_timing', 'text_without_alignment', 'alignment_failed'}


def events(acoustic, threshold=.1, use_level=True, minimum=.15):
    """Fixed hypotheses, not calibrated confidence. Preserve the earliest event."""
    step = acoustic['frame_seconds']
    scores, db = acoustic['curves']['vocal'], acoustic['rms_db']
    if not qwen.finite(step) or step <= 0 or len(scores) != len(db) or not scores:
        raise ValueError('Invalid acoustic timeline')
    if any(not qwen.finite(v) for v in scores+db):
        raise ValueError('Non-finite acoustic evidence')
    # Compare with/without this gate explicitly. No stem normalization; a
    # relative threshold can still miss quiet vocals and is not ground truth.
    floor = max(-85., max(db)-40.)
    mask = [s >= threshold and (not use_level or d >= floor) for s, d in zip(scores, db)]
    ranges, start = [], None
    for i, active in enumerate(mask+[False]):
        if active and start is None:
            start = i
        if not active and start is not None:
            if (i-start)*step >= minimum-1e-9:
                ranges.append([round(start*step, 4), round(min(i*step, acoustic['duration']), 4)])
            start = None
    return ranges


def near_event(cue, ranges, slack=.75):
    return next((start for start, end in ranges if start-slack <= cue <= end+slack), None)


def word_value(row, strict=False):
    if not row:
        return None
    flags = set(row.get('quality_flags', []))
    if flags & BAD_TIMING or (strict and flags - {'nonlexical_tokens_ignored'}):
        return None
    value = row.get('first_word_start')
    return float(value) if qwen.finite(value) and 0 <= value < row['duration'] else None


def text_agrees(left, right):
    # Agreement is one ablation, never independent proof. Keep exclamations.
    def tokens(row):
        return set(re.findall(r'[^\W_]+', row.get('text', '').casefold())[:5])
    a, b = tokens(left), tokens(right)
    return bool(a and b and len(a & b)/len(a | b) >= .5)


def smad_event(smad, threshold, consecutive):
    """First run of clip-level SMAD singing scores; returns its first window."""
    if not smad or smad.get('class_order') is None:
        return None
    try:
        index = smad['class_order'].index('singing_music')
        starts, probs = smad['window_starts'], smad['probabilities']
        if len(starts) != len(probs):
            return None
        active = [qwen.finite(row[index]) and row[index] >= threshold for row in probs]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    run = 0
    for i, yes in enumerate(active):
        run = run + 1 if yes else 0
        if run >= consecutive:
            return float(starts[i-run+1])
    return None


def candidates(assets, asr, acoustics, smad=None):
    output = []
    def add(chain, value, reason, required):
        output.append(dict(chain=chain, cue_s=None if value is None else round(value, 4),
                           state='abstain' if value is None else 'candidate', reason=reason, required=required))
    for variant in INPUTS:
        for model in ('turbo', 'qwen', 'whisper_auto'):
            row = asr.get((variant, model))
            model_label = 'whisper-auto' if model == 'whisper_auto' else model
            value = word_value(row)
            add(f'{variant}/{model_label}/words', value, 'Valid first word; text may be hallucinated', [variant, model_label])
            add(f'{variant}/{model_label}/quality', word_value(row, True), 'ASR flags gate', [variant, model_label])
            a = acoustics.get((variant, 4))
            level_ok = False
            if a and value is not None:
                lo, hi = max(0, int((value-.15)/.01)), min(len(a['rms_db']), int((value+.35)/.01)+1)
                level_ok = bool(a['rms_db'][lo:hi] and max(a['rms_db'][lo:hi]) >= max(-85, max(a['rms_db'])-40))
            add(f'{variant}/{model_label}/level', value if level_ok else None,
                'First word plus local level; level is not vocal identity', [variant, model_label, 'levels'])
            ratio_ok = bool(a and value is not None and a.get('stem_to_mix_db') and
                            max(a['stem_to_mix_db'][lo:hi], default=-200) >= -30)
            add(f'{variant}/{model_label}/level+ratio', value if level_ok and ratio_ok else None,
                'Level plus stem/mix ratio >= -30 dB near first word; experimental quiet-vocal risk',
                [variant, model_label, 'levels', 'original'])
            for context in (4, 8):
                a = acoustics.get((variant, context))
                ranges = events(a) if a else []
                onset = near_event(value, ranges) if value is not None else None
                # A later lyric cannot erase an earlier acoustic vocal candidate.
                earlier = bool(ranges and onset is not None and ranges[0][0] < onset-.75)
                gated = onset if onset is not None and not earlier and word_value(row, True) is not None else None
                add(f'{variant}/{model_label}/vocal-{context}s', gated,
                    'Earlier vocal evidence conflicts' if earlier else 'ASR plus earliest level/PANNs event; refine to event start',
                    [variant, model_label, f'panns-{context}s'])
        for context in (4, 8):
            a = acoustics.get((variant, context))
            for threshold in (.1, .3, .5):
                for gate in (False, True):
                    ranges = events(a, threshold, gate) if a else []
                    add(f'{variant}/panns-{context}s/{threshold}/{"level" if gate else "raw"}',
                        ranges[0][0] if ranges else None, 'Earliest acoustic vocal event; includes non-word vocals',
                        [variant, f'panns-{context}s'])
        left, right = asr.get((variant, 'turbo')), asr.get((variant, 'qwen'))
        l, r = word_value(left, True), word_value(right, True)
        agree = l is not None and r is not None and abs(l-r) <= .75 and text_agrees(left, right)
        add(f'{variant}/two-asr', min(l, r) if agree else None,
            'Two recognizers agree in time and opening tokens; may share hallucinations', [variant, 'turbo', 'qwen'])
        ranges = events(acoustics[(variant, 8)]) if (variant, 8) in acoustics else []
        onset = near_event(min(l, r), ranges) if agree else None
        good = onset is not None and ranges and abs(onset-ranges[0][0]) <= .75
        add(f'{variant}/two-asr+vocal', onset if good else None,
            'Two recognizers plus earliest acoustic event', [variant, 'turbo', 'qwen', 'panns-8s'])
    # A second separator agreeing does not imply statistical independence.
    for context in (4, 8):
        first = {v: events(acoustics[(v, context)]) for v in INPUTS if (v, context) in acoustics}
        times = {v: spans[0][0] for v, spans in first.items() if spans}
        supported = [t for v, t in times.items() if any(w != v and abs(u-t) <= .75 for w, u in times.items())]
        value = min(supported) if supported else None
        # Any earlier unmatched event blocks this consensus variant, rather
        # than silently choosing the later, more convenient lyric.
        if value is not None and any(t < value-.75 for t in times.values()):
            value = None
        if len(first) != len(INPUTS):
            value = None
        add(f'ensemble/panns-{context}s', value, 'Earliest event corroborated by two inputs; all active inputs required',
            list(INPUTS)+[f'panns-{context}s'])

    # Keep cross-model consensus genuinely cross-model: Whisper's forced-English
    # and auto-language runs share weights and are not independent votes.
    for variant in INPUTS:
        qwen_row = asr.get((variant, 'qwen'))
        auto_row = asr.get((variant, 'whisper_auto'))
        left, right = word_value(qwen_row, True), word_value(auto_row, True)
        agree = (left is not None and right is not None and abs(left-right) <= .75
                 and text_agrees(qwen_row, auto_row))
        add(f'{variant}/qwen+whisper-auto', min(left, right) if agree else None,
            'Qwen and auto-language Whisper agree in opening tokens and within 750ms; still may hallucinate',
            [variant, 'qwen', 'whisper-auto'])

    # SMAD is a 4-second clip classifier, not a boundary detector. The tested
    # offsets make its localization uncertainty explicit rather than claiming
    # each positive window start is an onset.
    if smad:
        for threshold in (.35, .5, .65):
            for consecutive in (1, 2, 3):
                event = smad_event(smad, threshold, consecutive)
                for offset in (0, 1, 2, 3, 4):
                    value = event + offset if event is not None else None
                    if value is not None and value > smad.get('duration', 30):
                        value = None
                    add(f'smad/t{threshold:.2f}/run{consecutive}/offset{offset}', value,
                        'First positive 4s singing_music window plus tested offset; clip-level only, not frame-aligned',
                        ['smad-original-4s'])

        # Cross-check SMAD's coarse singing clip with the frame-level PANNs
        # curves. Report the PANNs onset only when it is temporally associated
        # with a positive SMAD window; keep each input variant visible.
        for smad_threshold in (.35, .5):
            for run_length in (1, 2):
                clip_start = smad_event(smad, smad_threshold, run_length)
                for variant in INPUTS:
                    for context in (4, 8):
                        acoustic = acoustics.get((variant, context))
                        for panns_threshold in (.1, .3, .5):
                            for level_gate in (False, True):
                                ranges = events(acoustic, panns_threshold, level_gate) if acoustic else []
                                nearby = [start for start, _ in ranges if clip_start is not None
                                          and clip_start-.75 <= start <= clip_start+4.25]
                                onset = min(nearby) if nearby else None
                                gate = 'level' if level_gate else 'raw'
                                add(f'{variant}/smad{smad_threshold:.2f}-run{run_length}+panns-{context}s-{panns_threshold:.1f}-{gate}',
                                    onset, 'PANNs event associated with a positive SMAD 4s clip; PANNs onset reported',
                                    ['smad-original-4s', variant, f'panns-{context}s'])

        # ASR first words stay separate by recognizer and input. SMAD can
        # corroborate singing presence, but cannot prove that a transcript is
        # correct or identify the exact start inside its 4s window.
        for variant in INPUTS:
            for model in ('turbo', 'whisper_auto', 'qwen'):
                row = asr.get((variant, model))
                word = word_value(row, True)
                for threshold in (.35, .5):
                    clip_start = smad_event(smad, threshold, 1)
                    agree = word is not None and clip_start is not None and clip_start-.75 <= word <= clip_start+4.25
                    label = 'whisper-auto' if model == 'whisper_auto' else model
                    add(f'{variant}/smad{threshold:.2f}+{label}/words', word if agree else None,
                        'Quality-passing first word falls within positive SMAD 4s clip; transcript remains unverified',
                        ['smad-original-4s', variant, label])

    return output


def merge_labels(existing, songs, references):
    if existing.get('schema', 1) != 1:
        raise ValueError('Unsupported annotation schema')
    entries = list(existing.get('tracks', []))
    indexed = {}
    for row in entries:
        key = (row['song'], row['sha256'])
        if key in indexed:
            raise ValueError('Duplicate recording in annotations')
        indexed[key] = row
        for field in LABEL_FIELDS:
            value = row.get(field)
            if value is not None and (not qwen.finite(value) or value < 0):
                raise ValueError(f'Invalid {field} for {row["song"]}')
        if row.get('split', 'development') not in ('development', 'holdout'):
            raise ValueError('Annotation split must be development or holdout')
        tolerance = row.get('tolerance_s', .5)
        if not qwen.finite(tolerance) or tolerance < 0 or tolerance > 5:
            raise ValueError('Annotation tolerance must be 0–5 seconds')
        if not isinstance(row.get('no_vocal_in_window', False), bool):
            raise ValueError('no_vocal_in_window must be boolean')
        if row.get('no_vocal_in_window') and row.get('first_vocal_s') is not None:
            raise ValueError('A no-vocal window cannot also have a first vocal within it')
        for verdict in row.get('verdicts', {}).values():
            if not isinstance(verdict, dict) or verdict.get('value') not in ('correct', 'incorrect'):
                raise ValueError('Verdicts require value, cue_s and policy')
            if 'cue_s' not in verdict or 'policy' not in verdict or (verdict['cue_s'] is not None and not qwen.finite(verdict['cue_s'])):
                raise ValueError('Invalid verdict cue/policy')
    for song in songs:
        key = (song['song'], song['sha256'])
        if key in indexed:
            continue
        reference = base.reference_for(references, *key)
        value = reference.get('vocal_start') if reference and reference.get('audio_matches') else None
        row = dict(song=key[0], sha256=key[1], first_vocal_s=value, first_word_s=None,
                   radio_handover_s=None, no_vocal_in_window=False, tolerance_s=.5,
                   split='development', notes=reference.get('note', '') if value is not None else '', verdicts={})
        entries.append(row)
    return dict(schema=1, coordinates='original file seconds', tracks=entries)


def score(rows, labels, target='first_vocal_s', split='all'):
    known = {(x['song'], x['sha256']): x for x in labels['tracks']}
    summaries, details = [], []
    chains = sorted({c['chain'] for row in rows for c in row['candidates']})
    for chain in chains:
        verdicts = []
        for r in rows:
            label = known.get((r['song'], r['sha256']), {})
            if split != 'all' and label.get('split', 'development') != split:
                continue
            verdict = label.get('verdicts', {}).get(chain, {})
            candidate = next((c for c in r['candidates'] if c['chain'] == chain), {})
            if verdict.get('policy') == POLICY and 'cue_s' in candidate and verdict.get('cue_s') == candidate['cue_s']:
                verdicts.append(verdict.get('value'))
        errors, losses, early, late, abstained, covered, judged, correct, no_intro_false, censored_count = [], [], [], 0, 0, 0, 0, 0, 0, 0
        for row in rows:
            label = known.get((row['song'], row['sha256']), {})
            if split != 'all' and label.get('split', 'development') != split:
                continue
            truth = label.get(target)
            censored = target == 'first_vocal_s' and label.get('no_vocal_in_window', False)
            if truth is None and not censored:
                continue
            judged += 1
            candidate = next((c for c in row['candidates'] if c['chain'] == chain), {})
            cue = candidate.get('cue_s')
            # Values beyond the bounded window are censored, not exact targets.
            window = row['duration']
            censored = censored or (truth is not None and truth >= window)
            truth = window if censored else truth
            delta = cue-truth if cue is not None else None
            tolerance = label.get('tolerance_s', .5)
            censored_count += int(censored)
            if censored:
                # The labeled onset is at/outside the observed excerpt. Even a
                # candidate exactly on its right boundary is not verified by
                # audio in the scan window, so it cannot count as a precise hit.
                if cue is None:
                    abstained += 1
                    outcome = 'censored_abstain'
                else:
                    covered += 1
                    outcome = 'censored_candidate'
                loss, delta = 0., None
            elif cue is None:
                abstained += 1
                loss, outcome = 30., 'abstain'
            else:
                covered += 1
                errors.append(abs(delta))
                early.append(max(0., -delta))
                unsafe = not censored and delta > tolerance
                late += int(unsafe)
                no_intro_false += int(truth <= .5 and cue >= 1 and unsafe)
                ok = abs(delta) <= tolerance
                correct += int(ok)
                outcome = 'late' if unsafe else 'within_tolerance' if ok else 'early'
                loss = abs(delta)+4*max(0., delta-tolerance)
            losses.append(loss)
            details.append(dict(song=row['song'], sha256=row['sha256'], chain=chain, target=target,
                                truth_s=truth, censored=censored, cue_s=cue, error_s=delta,
                                outcome=outcome, loss=loss, split=label.get('split', 'development')))
        ordered = sorted(errors)
        summaries.append(dict(chain=chain, target=target, split=split, labelled=judged, covered=covered,
            listening_correct=verdicts.count('correct'), listening_incorrect=verdicts.count('incorrect'),
            abstained=abstained, censored=censored_count, coverage=covered/judged if judged else None, within_tolerance=correct,
            goal_90_count=math.ceil(.9*judged) if target == 'first_vocal_s' and judged else None,
            goal_90_met=(correct >= math.ceil(.9*judged)) if target == 'first_vocal_s' and judged else None,
            correct_fraction_all=correct/judged if judged else None, unsafe_late=late,
            false_usable_intro=no_intro_false, mae_s=statistics.mean(errors) if errors else None,
            p90_abs_s=ordered[max(0, math.ceil(.9*len(ordered))-1)] if ordered else None,
            mean_lost_intro_s=statistics.mean(early) if early else None,
            weighted_loss=statistics.mean(losses) if losses else None))
    return sorted(summaries, key=lambda r: (r['weighted_loss'] is None, r['weighted_loss'] or 0, r['chain'])), details


def verified_evidence(args, root):
    matrix = read(root/'matrix.json')
    jobs, errors = qwen.select_jobs(args.evidence, args.songs, matrix, None, INPUTS, {})
    qreport = read(args.evidence/'qwen/report.json') if (args.evidence/'qwen/report.json').exists() else {'rows': []}
    areport = read(root/'acoustic-report.json') if (root/'acoustic-report.json').exists() else {'rows': []}
    lreport = read(root/'language/report.json') if (root/'language/report.json').exists() else {'rows': []}
    sreport = read(root/'smad-report.json') if (root/'smad-report.json').exists() else {'rows': []}
    rows = []
    for song in matrix['songs']:
        digest = base.sha256(qwen.beneath(args.songs, song))
        if matrix['fingerprints'][song] != digest:
            raise ValueError(f'Recording changed since matrix: {song}; rerun benchmark')
        assets, asr, acoustics, smad = {}, {}, {}, None
        for job in (j for j in jobs if j['song'] == song):
            variant = job['input']
            assets[variant] = job
            oldrow = next(r for r in matrix['rows'] if r['song'] == song and r['input'] == variant)
            raw = read(qwen.beneath(args.evidence, oldrow['transcript']))['result']
            summary = stems.describe_words(raw, job['duration'], None)
            asr[(variant, 'turbo')] = dict(summary, text=raw.get('text', ''), duration=job['duration'])
            for saved in lreport['rows']:
                if (saved.get('song'), saved.get('input'), saved.get('sha256'), saved.get('audio_hash')) != (song, variant, digest, job['audio_hash']):
                    continue
                try:
                    transcript = read(qwen.beneath(args.evidence, saved['transcript']))
                    identity = transcript['identity']
                    if (transcript['status'] != 'complete' or identity.get('profile') != 'whisper-turbo-auto-language-30s-v1'
                            or identity.get('song') != song or identity.get('sha256') != digest
                            or identity.get('input') != variant or identity.get('audio_hash') != job['audio_hash']
                            or abs(identity.get('duration', -99) - job['duration']) > .03
                            or identity.get('model') != 'turbo' or identity.get('options', {}).get('language') is not None):
                        raise ValueError('Whisper auto-language cache identity mismatch')
                    summary = stems.describe_words(transcript['result'], job['duration'], None)
                    asr[(variant, 'whisper_auto')] = dict(summary, text=transcript['result'].get('text', ''),
                        language=transcript['result'].get('language'), duration=job['duration'])
                except Exception as error:
                    errors.append(dict(song=song, input=variant, stage='read_whisper_auto', error=str(error)))
            for saved in qreport['rows']:
                if (saved['song'], saved['input'], saved.get('sha256'), saved.get('audio_hash')) != (song, variant, digest, job['audio_hash']):
                    continue
                try:
                    transcript = read(qwen.beneath(args.evidence/'qwen', saved['transcript']))
                    identity = transcript['identity']
                    if transcript['status'] != 'complete' or identity['audio_sha256'] != job['audio_hash'] or identity['duration'] != job['duration']:
                        raise ValueError('Qwen cache identity mismatch')
                    aligned = None
                    if saved.get('alignment'):
                        aligned = read(qwen.beneath(args.evidence/'qwen', saved['alignment']))
                        if aligned['identity']['asr'] != identity or aligned['identity']['result'] != transcript['result']:
                            raise ValueError('Qwen alignment identity mismatch')
                    summary = qwen.summarize(transcript['result'], aligned['result'] if aligned else None, job['duration'], None)
                    asr[(variant, 'qwen')] = dict(summary, duration=job['duration'])
                except Exception as error:
                    errors.append(dict(song=song, input=variant, stage='read_qwen', error=str(error)))
            for saved in areport['rows']:
                if (saved['song'], saved['input'], saved.get('sha256'), saved.get('audio_hash')) != (song, variant, digest, job['audio_hash']):
                    continue
                try:
                    record = read(qwen.beneath(root, saved['evidence']))
                    identity = record['identity']
                    if record['status'] != 'complete' or identity['audio_sha256'] != job['audio_hash'] or identity['duration'] != job['duration'] or identity['input_seconds'] != saved['input_seconds']:
                        raise ValueError('Acoustic cache identity mismatch')
                    acoustics[(variant, saved['input_seconds'])] = record['result']
                except Exception as error:
                    errors.append(dict(song=song, input=variant, stage='read_acoustic', error=str(error)))
            if (variant, 'qwen') not in asr:
                errors.append(dict(song=song, input=variant, stage='qwen', error='No matching Qwen evidence'))
            for context in (4, 8):
                if (variant, context) not in acoustics:
                    errors.append(dict(song=song, input=variant, stage=f'panns-{context}s', error='No matching acoustic evidence'))
        if song in {r.get('song') for r in sreport.get('rows', [])}:
            saved = next(r for r in sreport['rows'] if r.get('song') == song)
            try:
                record = read(qwen.beneath(root, saved['evidence']))
                identity = record['identity']
                if (record['status'] != 'complete' or identity.get('profile') != 'smad-crnn-4s-sliding-v2-first-30s'
                        or identity.get('song') != song or identity.get('audio_sha256') != digest
                        or abs(identity.get('duration', -99) - min(30., next(j['duration'] for j in jobs if j['song'] == song))) > .03
                        or identity.get('model', {}).get('revision') != sreport.get('model', {}).get('revision')
                        or record['result'].get('class_order') != ['speech_noise', 'speech_music', 'singing_music', 'none']):
                    raise ValueError('SMAD cache identity mismatch')
                smad = record['result']
            except Exception as error:
                errors.append(dict(song=song, stage='read_smad', error=str(error)))
        else:
            errors.append(dict(song=song, stage='smad', error='No matching SMAD evidence'))
        rows.append(dict(song=song, sha256=digest, duration=next(iter(assets.values()))['duration'] if assets else 30,
            assets=assets, asr={f'{v}/{m}': r for (v, m), r in asr.items()},
            acoustic={f'{v}/{n}': a for (v, n), a in acoustics.items()},
            smad=smad, candidates=candidates(assets, asr, acoustics, smad)))
    return rows, errors


def tsv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.with_suffix('.tmp').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix('.tmp').replace(path)


def build_review(root, rows, labels):
    review = root/'review'
    (review/'audio').mkdir(parents=True, exist_ok=True)
    data = []
    for row in rows:
        audio = {}
        for variant, asset in row['assets'].items():
            dest = review/'audio'/f'{asset["audio_hash"]}.wav'
            if not dest.exists() or base.sha256(dest) != asset['audio_hash']:
                shutil.copyfile(asset['audio'], dest)
            audio[variant] = f'audio/{dest.name}'
        data.append(dict(song=row['song'], sha256=row['sha256'], duration=row['duration'], audio=audio,
                         candidates=row['candidates'], asr=row['asr'], acoustic=row['acoustic'], smad=row.get('smad')))
    template = Path(__file__).with_name('intro_review.html').read_text()
    payload = json.dumps(dict(rows=data, labels=labels, policy=POLICY), ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (review/'index.html').write_text(template.replace('/*BENCHMARK_DATA*/null', payload))


def timing_rows(evidence):
    """Report observed stage costs; cached zeroes never count as fast inference."""
    output = []
    for relative in ('comparison-report.json', 'ultimate/acoustic-report.json',
                     'qwen/report.json', 'ultimate/language/report.json', 'ultimate/smad-report.json'):
        if not (evidence/relative).exists():
            continue
        report = read(evidence/relative)
        groups = {}
        for row in report.get('rows', []):
            kind = ('smad-4s-sliding' if relative.endswith('/smad-report.json') else
                    'whisper-auto-language' if relative.endswith('/language/report.json') else
                    'panns-'+str(row['input_seconds']) if 'input_seconds' in row else
                    'qwen+alignment' if 'asr_cached' in row else 'whisper-turbo')
            # SMAD v2 reports omit the input field; the model always
            # analyzes the original mix. Accept those completed reports intact.
            variant = row.get('input', 'original') if kind == 'smad-4s-sliding' else row['input']
            key = (variant, kind)
            group = groups.setdefault(key, dict(fresh_jobs=0, cached_jobs=0, seconds_this_report=0., fresh_seconds=0.))
            cached = (row.get('asr_cached') and row.get('alignment_cached')) if 'asr_cached' in row else row.get('reused', False)
            group['cached_jobs' if cached else 'fresh_jobs'] += 1
            seconds = row.get('seconds_this_run', row.get('asr_seconds_this_run', 0)+row.get('alignment_seconds_this_run', 0))
            # Old SMAD cache hits recorded the original inference duration.
            if kind == 'smad-4s-sliding' and cached:
                seconds = 0.
            group['seconds_this_report'] += seconds
            if not cached:
                group['fresh_seconds'] += seconds
        for row in report.get('audio_preparation', []):
            key = (row.get('input', 'original'), 'separation')
            group = groups.setdefault(key, dict(fresh_jobs=0, cached_jobs=0, seconds_this_report=0., fresh_seconds=0.))
            group['cached_jobs' if row.get('reused') else 'fresh_jobs'] += 1
            seconds = row.get('prepare_seconds_this_run', 0)
            group['seconds_this_report'] += seconds
            if not row.get('reused'):
                group['fresh_seconds'] += seconds
        for (variant, stage), group in groups.items():
            output.append(dict(source_report=relative, input=variant, stage=stage, **group,
                mean_fresh_job_s=group['fresh_seconds']/group['fresh_jobs'] if group['fresh_jobs'] else None))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('/workspace/experiments/whisper-evidence'))
    parser.add_argument('--songs', type=Path, default=Path('/workspace/input/songs'))
    parser.add_argument('--references', type=Path, default=Path(__file__).resolve().parents[1]/'experiments/vocal-references.json')
    parser.add_argument('--labels', type=Path, help='Import listening annotations exported by review page')
    parser.add_argument('--target', choices=LABEL_FIELDS, default='first_vocal_s')
    parser.add_argument('--no-review', action='store_true', help='Skip portable audio copies/HTML')
    args = parser.parse_args()
    args.evidence, args.songs = args.evidence.resolve(), args.songs.resolve()
    if args.evidence.is_relative_to(args.songs) or args.songs.is_relative_to(args.evidence):
        parser.error('Input and output trees must be separate')
    root = root_for(args.evidence)
    with (args.evidence/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, errors = verified_evidence(args, root)
        label_path = args.labels or root/'annotations.json'
        labels = merge_labels(read(label_path) if label_path.exists() else {}, rows,
                              read(args.references) if args.references.exists() else {})
        if args.labels and not args.labels.exists():
            raise ValueError('Requested annotation file does not exist')
        # Preserve the previous annotation snapshot before an explicit import.
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        if args.labels and (root/'annotations.json').exists():
            shutil.copyfile(root/'annotations.json', root/'runs'/f'annotations-before-{stamp}.json')
        base.atomic_json(root/'annotations.json', labels)
        ranking, details = score(rows, labels, args.target)
        development, _ = score(rows, labels, args.target, 'development')
        holdout, _ = score(rows, labels, args.target, 'holdout')
        report = dict(policy=POLICY, target=args.target, songs=len(rows), ranking=ranking,
            development=development, holdout=holdout, details=details, errors=errors,
            labels_sha256=base.sha256(root/'annotations.json'),
            notice='Experimental candidates only; no production markers. No model is automatically selected.',
            loss_definition='abs(error) + 4*max(0,error-tolerance); uncensored abstention/failure=30; censored abstention=0; lower is better',
            timing_notice='Stage reports distinguish cache hits/load/inference. Do not compare resumed runs as fresh throughput.',
            timings=timing_rows(args.evidence), rows=rows)
        base.atomic_json(root/'report.json', report)
        base.atomic_json(root/'runs'/f'evaluation-{stamp}.json', report)
        tsv(root/'ranking.tsv', ranking)
        tsv(root/'development.tsv', development)
        tsv(root/'holdout.tsv', holdout)
        tsv(root/'details.tsv', details)
        tsv(root/'timings.tsv', report['timings'])
        if not args.no_review:
            build_review(root, rows, labels)
        print(f'Review: {root/"review/index.html"}\nRanking: {root/"ranking.tsv"}\nAnnotations: {root/"annotations.json"}')
        print(f'{len(rows)} recordings; {ranking[0]["labelled"] if ranking else 0} labelled for {args.target}; {len(errors)} evidence errors')


if __name__ == '__main__':
    main()

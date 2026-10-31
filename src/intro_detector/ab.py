"""A/B comparison of a pipeline change on a fixed song set.

A is the stored production analysis, B a fresh Pod analysis with the current pod/.
B evidence goes to the separate ab_analyses table: nothing is published to AzuraCast
and production analyses, decisions and publications stay untouched.
"""
from collections import Counter, defaultdict
import csv
import dataclasses
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import time
import uuid
import zlib

from . import evidence
from .db import pack, unpack

NAME = re.compile(r'[A-Za-z0-9_-]{1,64}')
# Language that Whisper detected on the Mel-Band stem: the per-song ASR language of B.
LANGUAGE = re.compile(rb'"melband/whisper_auto":\{.*?"language":(null|"[^"]*")', re.S)
CHANGED_SECONDS = 0.5


def _paths(data_dir, name):
    if not NAME.fullmatch(name):
        raise ValueError('Set name: letters, digits, "-" and "_" only')
    directory = Path(data_dir) / 'ab'
    return directory / f'{name}.json', directory / f'{name}-report.tsv'


def stored_language(blob):
    data = zlib.decompress(blob)
    match = LANGUAGE.search(data)
    if match:
        return json.loads(match.group(1))
    return (json.loads(data)['asr'].get('melband/whisper_auto') or {}).get('language')


def _language_index(con, data_dir, live, log):
    """{sha256: language} of all live analyses; cached because reading every report takes minutes."""
    path = Path(data_dir) / 'ab' / 'languages.json'
    count = con.execute('SELECT COUNT(*) FROM analyses').fetchone()[0]
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('analyses') == count:
            return {sha: lang for sha, lang in cached['languages'].items() if sha in live}
    log(f'Reading the stored language of {count} analyses (takes several minutes)')
    index = {sha: stored_language(blob) for sha, blob in con.execute('SELECT sha256, report FROM analyses')}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(analyses=count, languages=index)))
    return {sha: lang for sha, lang in index.items() if sha in live}


def select(con, data_dir, name, count=80, controls=10, seed=27, quotas=None, log=print):
    """Non-English recordings plus English controls.

    quotas ({language: n}, 'other' for all unlisted languages) fixes the mix; without
    quotas, count recordings are taken round-robin across languages.
    """
    set_path, _ = _paths(data_dir, name)
    if set_path.exists():
        raise ValueError(f'Set {name} already exists: {set_path}')
    live = {}
    for row in con.execute("SELECT * FROM media WHERE status='done' AND sha256 IS NOT NULL "
                           'ORDER BY station_id, media_id'):
        live.setdefault(row['sha256'], row)
    languages, english, unknown = defaultdict(list), [], 0
    for sha, language in sorted(_language_index(con, data_dir, live, log).items()):
        if language is None:
            unknown += 1
        elif language == 'en':
            english.append(sha)
        else:
            languages[language].append(sha)
    rng = random.Random(seed)
    for shas in languages.values():
        rng.shuffle(shas)
    order = sorted(languages, key=lambda code: (-len(languages[code]), code))
    available = Counter({code: len(shas) for code, shas in languages.items()})
    chosen = []
    if quotas:
        others = [(sha, code) for code in order if code not in quotas for sha in languages[code]]
        rng.shuffle(others)
        for code, n in quotas.items():
            pool = others if code == 'other' else [(sha, code) for sha in languages.get(code, [])]
            chosen += [(sha, language, False) for sha, language in pool[:n]]
    else:
        while len(chosen) < count and any(languages[code] for code in order):
            for code in order:
                if languages[code] and len(chosen) < count:
                    chosen.append((languages[code].pop(), code, False))
    chosen += [(sha, 'en', True) for sha in rng.sample(english, min(controls, len(english)))]
    entries = [dict(sha256=sha, language=code, control=control, station_id=live[sha]['station_id'],
                    media_id=live[sha]['media_id'], path=live[sha]['path']) for sha, code, control in chosen]
    set_path.parent.mkdir(parents=True, exist_ok=True)
    set_path.write_text(json.dumps(dict(name=name, created_at=time.time(), seed=seed, quotas=quotas,
                                        entries=entries), indent=1))
    picked = Counter(e['language'] for e in entries if not e['control'])
    log(f'Available non-English: {sum(available.values())} '
        f'({", ".join(f"{k} {available[k]}" for k in order)}); English {len(english)}; unknown {unknown}')
    log(f'Selected {sum(picked.values())} non-English ({", ".join(f"{k} {v}" for k, v in picked.most_common())}) '
        f'and {len(entries) - sum(picked.values())} English controls -> {set_path}')
    return entries


def load(data_dir, name):
    set_path, _ = _paths(data_dir, name)
    if not set_path.exists():
        raise ValueError(f'Unknown set {name}; run ab-select first')
    return json.loads(set_path.read_text())['entries']


def cost_limits(settings, spent, max_usd):
    """Runpod settings that keep one run below max_usd at the hourly price cap.

    Billed time is Pod start, the batch and the idle period before deletion, so the
    batch may run at most max_usd / price minus startup and idle time.
    """
    billable = max_usd / settings.max_price_per_hour * 60
    minutes = int(billable - settings.startup_minutes - settings.idle_seconds / 60 + 1e-9)
    if minutes < 10:
        raise ValueError(f'{max_usd:.2f} USD buys only {billable:.0f} GPU minutes at '
                         f'{settings.max_price_per_hour:.2f} USD/h; too little for a batch')
    return dataclasses.replace(settings, monthly_budget_usd=spent + max_usd, max_run_minutes=minutes)


def run(service, name, max_usd=None, log=print):
    """Analyze the set's recordings that have no B evidence yet. Never publishes."""
    con, entries = service.con, load(service.cfg.data_dir, name)
    done = {r[0] for r in con.execute('SELECT sha256 FROM ab_analyses WHERE set_name=?', (name,))}
    todo = [e for e in entries if e['sha256'] not in done]
    result = dict(already=len(entries) - len(todo), stored=0, failed={}, changed_audio=[])
    if not todo:
        return result
    if max_usd is not None:
        service.gpu.cfg = cost_limits(service.gpu.cfg, service.gpu.spend(), max_usd)
        log(f'Cost cap {max_usd:.2f} USD: batch stops after {service.gpu.cfg.max_run_minutes} minutes')
    batch = service.cfg.data_dir / 'work' / f'ab-{uuid.uuid4().hex}'
    songs = batch / 'songs'
    try:
        wanted = set()
        for entry in todo:
            suffix = PurePosixPath(entry['path']).suffix.lower()
            path, sha = service.client.download(entry['station_id'], entry['media_id'], songs, suffix)
            if sha != entry['sha256']:
                path.unlink(missing_ok=True)
                result['changed_audio'].append(entry['path'])
                continue
            wanted.add(sha)
        if not wanted:
            return result
        log(f'Analyzing {len(wanted)} recordings on the GPU (set {name})')
        service.gpu.run(songs, batch / 'reports')
        report, music = evidence.load_reports(batch / 'reports/report.json', batch / 'reports/music-report.json')
        complete, failed = evidence.split(report, music, wanted)
        with con:
            for sha, (row, tune) in complete.items():
                con.execute('INSERT OR REPLACE INTO ab_analyses VALUES(?,?,?,?,?)',
                            (name, sha, pack(row), pack(tune), time.time()))
        result.update(stored=len(complete), failed=failed)
        return result
    finally:
        shutil.rmtree(batch, ignore_errors=True)


def _first(row, key):
    return (row['asr'].get(key) or {}).get('first_word_start')


def _text(row, key):
    return ' '.join(str((row['asr'].get(key) or {}).get('text') or '').split())[:80]


def _changed(a, b):
    if a is None or b is None:
        return a is not b
    return abs(a - b) > CHANGED_SECONDS


def _station_names(service):
    try:
        return service.client.station_names()
    except Exception:
        return {}


def report(service, name, log=print):
    """TSV with A and B decisions per recording plus a summary; evaluates without storing."""
    con, entries = service.con, load(service.cfg.data_dir, name)
    _, tsv = _paths(service.cfg.data_dir, name)
    names = _station_names(service)
    rows, summary = [], defaultdict(Counter)
    for entry in entries:
        group = 'control' if entry['control'] else 'non-English'
        a = con.execute('SELECT report, music FROM analyses WHERE sha256=?', (entry['sha256'],)).fetchone()
        b = con.execute('SELECT report, music FROM ab_analyses WHERE set_name=? AND sha256=?',
                        (name, entry['sha256'])).fetchone()
        if a is None or b is None:
            summary[group]['missing'] += 1
            continue
        ra, rb = evidence.clean(unpack(a['report'])), evidence.clean(unpack(b['report']))
        _, ma, va = service.evaluate(ra, unpack(a['music']), entry['sha256'])
        _, mb, vb = service.evaluate(rb, unpack(b['music']), entry['sha256'])
        changed = _changed(va, vb)
        summary[group]['songs'] += 1
        summary[group]['changed' if changed else 'same'] += 1
        if va is None and vb is not None:
            summary[group]['newly resolved'] += 1
        if va is not None and vb is None:
            summary[group]['newly unresolved'] += 1
        rows.append([group, entry['language'], (rb['asr'].get('melband/qwen') or {}).get('language'),
                     entry['station_id'], names.get(entry['station_id'], ''), entry['media_id'], entry['path'],
                     ma.get('intro_type') or ma.get('intro_reason'), va,
                     mb.get('intro_type') or mb.get('intro_reason'), vb,
                     None if va is None or vb is None else round(vb - va, 2), int(changed),
                     _first(ra, 'melband/turbo'), _first(rb, 'melband/turbo'),
                     _first(ra, 'melband/qwen'), _first(rb, 'melband/qwen'),
                     _text(ra, 'melband/turbo'), _text(rb, 'melband/turbo')])
    rows.sort(key=lambda r: (r[0] != 'non-English', -r[12], r[1], r[6]))
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with tsv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['group', 'language', 'b_qwen_language', 'station', 'station_name', 'media', 'path',
                         'a_type', 'a_intro_end', 'b_type', 'b_intro_end', 'delta_s', 'changed',
                         'a_whisper_melband_s', 'b_whisper_melband_s', 'a_qwen_melband_s', 'b_qwen_melband_s',
                         'a_whisper_text', 'b_whisper_text'])
        writer.writerows(rows)
    for group, counts in sorted(summary.items()):
        log(f'{group}: ' + ', '.join(f'{k} {v}' for k, v in counts.items()))
    for row in rows:
        if row[12]:
            log(f'  changed: {row[4] or "station " + str(row[3])} | media {row[5]} | {row[6]} | '
                f'A {row[7]} {row[8]} -> B {row[9]} {row[10]}')
    log(f'Details: {tsv} (changed means intro_end differs by more than {CHANGED_SECONDS}s or one side is blank)')
    return {group: dict(counts) for group, counts in summary.items()}

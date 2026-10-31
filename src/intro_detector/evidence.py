"""Validate Pod reports and derive the selector's compact evidence rows."""
import copy
import json
import re

INPUTS = ('original', 'demucs', 'melband')
ASR_MODELS = ('turbo', 'qwen', 'whisper_auto')
REPORT_POLICY = 'intro-benchmark-v2-smad-auto-language'
MUSIC_PROFILE = 'percussion-energy-30s-v1'
CLEANING = 'subtitle-credits-v1'
# Subtitle/caption credits that Whisper invents in silence or for unfamiliar audio; never lyrics.
CREDITS = re.compile(
    r"sous-?titr|untertitel|subt[ií]tul|sottotitol|legendas? (?:pela|por)|ondertitel|napisy|"
    r"amara\.org|subtitles? by|ご視聴|시청해\s*주셔서|字幕", re.I)


def load_reports(report_path, music_path):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    music = json.loads(music_path.read_text(encoding='utf-8'))
    if report.get('policy') != REPORT_POLICY:
        raise ValueError(f'Unexpected report policy {report.get("policy")!r}')
    if music.get('profile') != MUSIC_PROFILE:
        raise ValueError(f'Unexpected music profile {music.get("profile")!r}')
    return report, music


def missing(row, music_row, errors=()):
    """Every model output is required; partial evidence could weaken a vocal veto."""
    problems = [f'{key} missing' for key in
                [f'{v}/{m}' for v in INPUTS for m in ASR_MODELS] if key not in row.get('asr', {})]
    problems += [f'acoustic {v}/{n} missing' for v in INPUTS for n in (4, 8)
                 if f'{v}/{n}' not in row.get('acoustic', {})]
    if not row.get('smad'):
        problems.append('smad missing')
    if music_row is None:
        problems.append('musical timing missing')
    problems += [f'{e.get("stage", "error")}: {e.get("error")}' for e in errors]
    return problems


def split(report, music, wanted):
    """Return {sha256: (report_row, music_row)} for complete rows and {sha256: problems} otherwise."""
    rows = {r['sha256']: r for r in report['rows'] if r.get('sha256') in wanted}
    tunes = {r['sha256']: r for r in music['rows'] if r.get('sha256') in wanted}
    songs = {r['song']: r['sha256'] for r in report['rows']}
    errors = {}
    for error in report.get('errors', []) + music.get('errors', []):
        sha = songs.get(error.get('song'))
        if sha in wanted:
            errors.setdefault(sha, []).append(error)
    complete, failed = {}, {}
    for sha in wanted:
        row = rows.get(sha)
        if row is None:
            failed[sha] = ['no report row']
            continue
        problems = missing(row, tunes.get(sha), errors.get(sha, ()))
        if problems:
            failed[sha] = problems
        else:
            complete[sha] = ({k: v for k, v in row.items() if k != 'assets'}, tunes[sha])
    return complete, failed


def _clean_asr(result):
    """Drop an invented credit and everything after it; None when nothing was found."""
    text = str(result.get('text') or '')
    words = list(result.get('words') or [])
    joined, offsets = '', []
    for word in words:
        offsets.append(len(joined))
        joined += str(word.get('word', ''))
    in_words, in_text = CREDITS.search(joined), CREDITS.search(text)
    if not (in_words or in_text):
        return None
    cleaned = dict(result, removed_credit=(in_words or in_text).group(0))
    if in_words:
        words = [w for w, start in zip(words, offsets) if start + len(str(w.get('word', ''))) <= in_words.start()]
    if in_text:
        text = text[:in_text.start()].strip()
    if not text:
        words = []
    cleaned.update(text=text, words=words, word_count=len(words),
                   first_words=''.join(str(w.get('word', '')) for w in words[:8]).strip())
    if result.get('first_word_start') is not None:
        starts = [w.get('start_original', w.get('start')) for w in words]
        cleaned['first_word_start'] = starts[0] if starts else None
    return cleaned


def clean(row):
    """Report row without invented subtitle credits in any ASR result; unchanged rows are returned as is."""
    changed = {name: cleaned for name, result in row.get('asr', {}).items()
               if (cleaned := _clean_asr(result)) is not None}
    if not changed:
        return row
    row = copy.copy(row)
    row['asr'] = {**row['asr'], **changed}
    return row


def compact(row):
    """Per-row fields the selector uses, without labels."""
    asr = {}
    for name, result in row['asr'].items():
        asr[name] = {field: result.get(field) for field in
                     ('first_word_start', 'quality_flags', 'language', 'duration')}
        asr[name]['opening_text'] = result.get('text', '')[:160]
    acoustic = {}
    for name, result in row['acoustic'].items():
        acoustic[name] = {field: result.get(field) for field in
                          ('duration', 'frame_seconds', 'rms_db', 'stem_to_mix_db', 'windows')}
        acoustic[name]['curves'] = {field: result['curves'].get(field) for field in
                                    ('vocal', 'singing', 'speech')}
    return dict(song=row['song'], sha256=row['sha256'], duration=row['duration'],
                label={'first_vocal_s': None, 'split': 'holdout'},
                candidates=[{field: candidate.get(field) for field in
                             ('chain', 'cue_s', 'state', 'reason', 'required')}
                            for candidate in row['candidates']],
                asr=asr, acoustic=acoustic,
                smad={field: (row.get('smad') or {}).get(field) for field in
                      ('duration', 'class_order', 'window_starts', 'probabilities')})

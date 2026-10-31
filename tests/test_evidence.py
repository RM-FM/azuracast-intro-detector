import unittest

from helpers import music_row, report_row
from intro_detector import evidence


class Split(unittest.TestCase):
    def test_complete_rows_drop_pod_paths(self):
        report = dict(rows=[report_row('a.mp3', 'a')], errors=[])
        complete, failed = evidence.split(report, dict(rows=[music_row('a.mp3', 'a')], errors=[]), {'a'})
        self.assertEqual(failed, {})
        self.assertNotIn('assets', complete['a'][0])

    def test_missing_model_output_or_stage_error_is_incomplete(self):
        row = report_row('a.mp3', 'a')
        del row['asr']['melband/whisper_auto']
        report = dict(rows=[row, report_row('b.mp3', 'b')],
                      errors=[dict(song='b.mp3', stage='smad', error='No matching SMAD evidence')])
        complete, failed = evidence.split(report, dict(rows=[music_row('a.mp3', 'a')], errors=[]), {'a', 'b', 'c'})
        self.assertEqual(complete, {})
        self.assertIn('melband/whisper_auto missing', failed['a'])
        self.assertIn('musical timing missing', failed['b'])
        self.assertIn('smad: No matching SMAD evidence', failed['b'])
        self.assertEqual(failed['c'], ['no report row'])

    def test_compact_matches_export_shape(self):
        row = report_row('a.mp3', 'a')
        row['asr']['original/turbo']['text'] = 'x' * 200
        row['acoustic']['original/4'] = {'frame_seconds': .01, 'curves': {'vocal': [0.], 'music': [1.]}}
        out = evidence.compact(row)
        self.assertEqual(len(out['asr']['original/turbo']['opening_text']), 160)
        self.assertEqual(out['acoustic']['original/4']['curves'], {'vocal': [0.], 'singing': None, 'speech': None})
        self.assertEqual(out['label'], {'first_vocal_s': None, 'split': 'holdout'})


def words(*pairs):
    return [dict(word=w, start=s, start_original=s) for w, s in pairs]


class Credits(unittest.TestCase):
    def asr(self, text, timed, first=None):
        return dict(text=text, words=words(*timed), first_word_start=first if first is not None else
                    (timed[0][1] if timed else None), quality_flags=[])

    def test_credit_only_transcript_has_no_first_word(self):
        row = report_row('a.mp3', 'a')
        row['asr']['melband/turbo'] = self.asr(' Sous-titres par Jérémy Diaz',
                                               [(' Sous', 29.04), ('-titres', 29.28), (' par', 29.52)])
        cleaned = evidence.clean(row)
        result = cleaned['asr']['melband/turbo']
        self.assertEqual((result['text'], result['words'], result['first_word_start']), ('', [], None))
        self.assertEqual(result['removed_credit'], 'Sous-titr')
        self.assertEqual(row['asr']['melband/turbo']['first_word_start'], 29.04)

    def test_lyrics_before_a_trailing_credit_are_kept(self):
        row = report_row('a.mp3', 'a')
        row['asr']['original/turbo'] = self.asr(" Ich treib mein Leben Untertitel im Auftrag des ZDF",
                                                [(' Ich', 3.5), (' treib', 3.9), (' Untertitel', 25.), (' im', 25.4)])
        result = evidence.clean(row)['asr']['original/turbo']
        self.assertEqual(result['first_word_start'], 3.5)
        self.assertEqual(result['text'], 'Ich treib mein Leben')
        self.assertEqual([w['word'] for w in result['words']], [' Ich', ' treib'])

    def test_invalid_first_word_stays_invalid_and_lyrics_untouched(self):
        row = report_row('a.mp3', 'a')
        row['asr']['demucs/qwen'] = self.asr('Subtítulos realizados por la comunidad de Amara.org',
                                             [('Subtítulos', 0.)], first=None)
        row['asr']['demucs/qwen']['first_word_start'] = None
        row['asr']['original/qwen'] = self.asr('Weil ich dir begegne', [('Weil', 9.3)])
        cleaned = evidence.clean(row)
        self.assertIsNone(cleaned['asr']['demucs/qwen']['first_word_start'])
        self.assertIs(cleaned['asr']['original/qwen'], row['asr']['original/qwen'])
        clean_row = report_row('b.mp3', 'b')
        self.assertIs(evidence.clean(clean_row), clean_row)


if __name__ == '__main__':
    unittest.main()

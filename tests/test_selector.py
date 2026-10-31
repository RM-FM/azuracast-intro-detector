import unittest

import helpers  # noqa: F401  (adds src to sys.path)

try:
    import sklearn  # noqa: F401
except ImportError:
    sklearn = None


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class FrozenSelector(unittest.TestCase):
    def test_inactive_input_cannot_supply_lead_words(self):
        from intro_detector.selector import _lead_word_events

        row = helpers.report_row('song.mp3', 'sha')
        row['asr']['bsroformer/turbo'] = dict(
            text='Hello there', words=[dict(word=' Hello', start_original=7.)],
            first_word_start=7., quality_flags=[])
        self.assertEqual(_lead_word_events(row), [])

    def test_inactive_evidence_does_not_change_prediction(self):
        from intro_detector.selector import Selector

        selector = Selector()
        row = helpers.report_row('song.mp3', 'sha')
        row['acoustic'] = {}
        expected = selector.predict(row)
        for model in ('turbo', 'qwen', 'whisper_auto'):
            row['asr'][f'bsroformer/{model}'] = dict(
                text='Words from the start', first_word_start=0., quality_flags=[],
                words=[dict(word=word, start_original=start) for word, start in
                       (('Words', 0.), ('from', .4), ('the', .8), ('start', 1.2))])
        row['acoustic']['bsroformer/4'] = dict(
            frame_seconds=.1, curves={'vocal': [1.] * 300})
        row['candidates'].append(dict(chain='bsroformer/panns-4s/0.1/level', cue_s=19.))
        self.assertEqual(selector.predict(row), expected)

    def test_packaged_training_matches_vendored_policy(self):
        from intro_detector.selector import POLICY, Selector
        selector = Selector()
        self.assertEqual(POLICY, 'intro-selector-extratrees-three-inputs-v2+lead-vocal-guard-v1')
        self.assertEqual(selector.training_songs, 39)

    def test_lead_word_after_vocalization_guards_late_model_cue(self):
        from intro_detector.selector import Selector

        row = helpers.report_row('song.mp3', 'sha')
        syllables = [' Be', '-ba', '-ba', '-ba', '-da', '-boom']
        starts = [0., .44, .76, 1., 1.24, 1.4]
        words = [dict(word=word, start_original=start) for word, start in zip(syllables, starts)]
        words.append(dict(word=" You're", start_original=7.56))
        for variant in ('original', 'demucs', 'melband', 'bsroformer'):
            for model in ('turbo', 'whisper_auto'):
                row['asr'][f'{variant}/{model}'] = dict(
                    text='Be-ba-ba-ba-da-boom You are here', words=words,
                    first_word_start=0., quality_flags=[])
            for context in (4, 8):
                row['acoustic'][f'{variant}/{context}'] = dict(
                    frame_seconds=.5, rms_db=[-40.] * 60, stem_to_mix_db=[0.] * 60,
                    curves={'vocal': [0.] * 60, 'singing': [0.] * 60, 'speech': [0.] * 60})
        row['candidates'] = [dict(chain='original/panns-4s/0.1/level', cue_s=12.)]
        row['smad'] = dict(class_order=['none', 'singing_music'], window_starts=[7.5],
                           probabilities=[[.1, .65]], duration=30.)

        selector = Selector()

        class CandidateModel:
            def __init__(self, preferred):
                self.preferred = preferred

            def predict(self, features):
                return self._np.array([1. if abs(feature[0] - self.preferred) < .01 else .5
                                       for feature in features])

            _np = selector._np

        for preferred in (0., 12.):
            selector.model = CandidateModel(preferred)
            result = selector.predict(row)
            self.assertAlmostEqual(result['lead_anchor_s'], 7.56)
            self.assertEqual(result['lead_anchor_support'], 3)
            self.assertTrue(result['lead_guard_applied'])
            self.assertGreaterEqual(result['cue_s'], result['lead_anchor_s'] - 1.)
            self.assertLessEqual(result['cue_s'], result['lead_anchor_s'])

    def test_invalid_word_alignment_does_not_create_lead_anchor(self):
        from intro_detector.selector import _lead_word_events

        row = helpers.report_row('song.mp3', 'sha')
        for variant in ('original', 'demucs', 'melband', 'bsroformer'):
            row['asr'][f'{variant}/turbo'] = dict(
                words=[dict(word=' Hello', start_original=7.)],
                first_word_start=7., quality_flags=['invalid_word_timing'])

        self.assertEqual(_lead_word_events(row), [])

    def test_repeated_one_letter_fragment_is_not_a_lead_word(self):
        from intro_detector.selector import _lead_word_events

        row = helpers.report_row('song.mp3', 'sha')
        for variant in ('original', 'demucs', 'melband', 'bsroformer'):
            row['asr'][f'{variant}/qwen'] = dict(
                text='B b', words=[dict(word=' B', start_original=.16),
                                   dict(word=' b', start_original=.56)],
                first_word_start=.16, quality_flags=[])

        self.assertEqual(_lead_word_events(row), [])

    def test_generic_and_repetitive_asr_openings_are_not_lead_events(self):
        from intro_detector.selector import _lead_word_events

        row = helpers.report_row('song.mp3', 'sha')
        for variant in ('original', 'demucs', 'melband', 'bsroformer'):
            row['asr'][f'{variant}/turbo'] = dict(
                text='Thank you.', words=[dict(word=' Thank', start_original=0.),
                                          dict(word=' you.', start_original=.5)],
                first_word_start=0., quality_flags=[])
        row['asr']['demucs/whisper_auto'] = dict(
            text='Aki, aki, aki, aki, aki, aki, aki, aki, aki, aki',
            words=[dict(word=' Aki,', start_original=index * .5) for index in range(10)],
            first_word_start=0., quality_flags=[])

        self.assertEqual(_lead_word_events(row), [])

    def test_no_intro_requires_independent_early_vocal_support(self):
        from intro_detector.selector import _early_vocal_support

        row = helpers.report_row('song.mp3', 'sha')
        for variant in ('original', 'demucs', 'melband', 'bsroformer'):
            row['asr'][f'{variant}/turbo'] = dict(
                text='Thank you.', words=[dict(word=' Thank', start_original=0.)],
                first_word_start=0., quality_flags=[])
            row['acoustic'][f'{variant}/4'] = dict(
                frame_seconds=.1, curves={'vocal': [0.] * 300})
        compact = __import__('intro_detector.evidence', fromlist=['compact']).compact(row)

        self.assertFalse(_early_vocal_support(compact, row))
        row['acoustic']['original/4']['curves']['vocal'][:10] = [.2] * 10
        row['acoustic']['demucs/4']['curves']['vocal'][:10] = [.2] * 10
        compact = __import__('intro_detector.evidence', fromlist=['compact']).compact(row)
        self.assertTrue(_early_vocal_support(compact, row))

    def test_early_panns_conflict_needs_vocal_corroboration(self):
        from intro_detector.evidence import compact
        from intro_detector.selector import _early_panns_are_unverified

        row = helpers.report_row('song.mp3', 'sha')
        row['candidates'] = [dict(chain=f'{variant}/panns-4s/0.1/level', cue_s=0.)
                             for variant in ('original', 'demucs', 'melband')]
        compact_row = compact(row)

        self.assertTrue(_early_panns_are_unverified(compact_row, row, 29.))
        for variant in ('original', 'demucs'):
            row['acoustic'][f'{variant}/4'] = dict(
                frame_seconds=.1, curves={'vocal': [.2] * 10 + [0.] * 290})
        compact_row = compact(row)
        self.assertFalse(_early_panns_are_unverified(compact_row, row, 29.))

    def test_lyrics_from_the_start_are_early_vocals_without_acoustic_activity(self):
        from intro_detector.evidence import compact
        from intro_detector.selector import _early_vocal_support

        row = helpers.report_row('song.mp3', 'sha')
        lyric = [dict(word=w, start_original=s) for w, s in
                 ((" I've", 0.), (' never', .84), (' known', 1.2), (' someone', 1.8))]
        for variant in ('original', 'demucs', 'melband'):
            row['asr'][f'{variant}/turbo'] = dict(
                text="I've never known someone", words=lyric, first_word_start=0., quality_flags=[])
        self.assertTrue(_early_vocal_support(compact(row), row))

        stray = [dict(word=' Yeah', start_original=0.), dict(word=' you', start_original=24.),
                 dict(word=' know', start_original=24.5)]
        for variant in ('original', 'demucs', 'melband'):
            row['asr'][f'{variant}/turbo'] = dict(
                text='Yeah you know', words=stray, first_word_start=0., quality_flags=[])
        self.assertFalse(_early_vocal_support(compact(row), row))

    def test_single_weak_burst_is_ignored_but_repeated_or_strong_bursts_start_vocals(self):
        from intro_detector.selector import _early_vocal_onset

        self.assertIsNone(_early_vocal_onset([(8.28, 9.2, .17)]))
        self.assertEqual(_early_vocal_onset([(8.28, 9.2, .17), (12.6, 14., .18)]), 8.28)
        self.assertEqual(_early_vocal_onset([(12.6, 15., .18)]), 12.6)
        self.assertEqual(_early_vocal_onset([(12.6, 13.2, .45)]), 12.6)

    def test_early_vocal_conflict_picks_an_onset_instead_of_vetoing(self):
        from intro_detector.selector import Selector

        selector = Selector()

        class LateModel:
            _np = selector._np

            def predict(self, features):
                return self._np.array([1. if abs(feature[0] - 25.) < .01 else .1 for feature in features])

        def song(bursts):
            row = helpers.report_row('song.mp3', 'sha')
            row['candidates'] = [dict(chain=f'{v}/panns-4s/0.1/level', cue_s=8.3)
                                 for v in ('original', 'demucs', 'melband')]
            row['candidates'].append(dict(chain='original/panns-8s/0.1/level', cue_s=25.))
            for variant in ('original', 'demucs', 'melband', 'bsroformer'):
                curve = [0.] * 300
                for start, end, level in bursts:
                    curve[int(start * 10):int(end * 10)] = [level] * (int(end * 10) - int(start * 10))
                for context in (4, 8):
                    row['acoustic'][f'{variant}/{context}'] = dict(
                        frame_seconds=.1, rms_db=[-40.] * 300, stem_to_mix_db=[0.] * 300,
                        curves={'vocal': curve, 'singing': [0.] * 300, 'speech': [0.] * 300})
            return row

        selector.model = LateModel()
        repeated = selector.predict(song([(8.3, 9.3, .17), (12.6, 14., .18)]))
        self.assertFalse(repeated['conflict'])
        self.assertAlmostEqual(repeated['decision_s'], 8.3)

        single = selector.predict(song([(8.3, 9.3, .17)]))
        self.assertFalse(single['conflict'])
        self.assertEqual(single['decision_s'], 25.)

        unsupported = selector.predict(song([]))
        self.assertTrue(unsupported['conflict'])
        self.assertIsNone(unsupported['decision_s'])


if __name__ == '__main__':
    unittest.main()

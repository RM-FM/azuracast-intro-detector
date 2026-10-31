"""Pod-side per-song language helpers (pod/tools/song_language.py); no GPU or models."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pod' / 'tools'))
import song_language  # noqa: E402


class PipelineInputs(unittest.TestCase):
    def test_commands_use_only_supported_inputs(self):
        from types import SimpleNamespace
        import qwen_experiment
        import run_ultimate
        import ultimate_inference

        inputs = ('original', 'demucs', 'melband')
        self.assertEqual(qwen_experiment.INPUTS, inputs)
        self.assertEqual(ultimate_inference.INPUTS, inputs)
        args = SimpleNamespace(songs=Path('/songs'), evidence=Path('/evidence'),
                               stem_python='python', qwen_python='python', whisper_cache=Path('/cache'))
        commands = dict(run_ultimate.commands(args))
        self.assertEqual(len(commands), 8)
        self.assertNotIn('apply-40-listening-labels', commands)
        self.assertTrue(all('--references' not in command for command in commands.values()))
        command = commands['qwen']
        self.assertEqual(command[command.index('--inputs') + 1:command.index('--language')], list(inputs))
        self.assertNotIn('bsroformer', ' '.join(part for command in commands.values() for part in command))

    def test_stages_do_not_require_private_listening_labels(self):
        from types import SimpleNamespace
        import run_ultimate

        args = SimpleNamespace(songs=Path('/songs'), evidence=Path('/evidence'),
                               stem_python='python', qwen_python='python', whisper_cache=Path('/cache'))
        selected = [Path('/songs/neutral-a.wav'), Path('/songs/neutral-b.wav')]
        stages = dict(run_ultimate.stages_for(args, selected))
        self.assertEqual(len(stages), 8)
        for name in ('whisper-auto-language', 'smad-singing-classifier'):
            self.assertEqual(stages[name][-2:], ['--expected-count', '2'])


def attempt_factory(results):
    calls = []

    def attempt(language):
        calls.append(language)
        words, text, reused = results[language]
        return ({'language': language}, f'{language}.json',
                dict(result=dict(text=text), elapsed_seconds=2.), reused, dict(word_count=words))
    return attempt, calls


class Fallback(unittest.TestCase):
    def test_needs_fallback(self):
        self.assertTrue(song_language.needs_fallback('fr', 0, ''))
        self.assertTrue(song_language.needs_fallback('fr', 5, ' Sous-titres par Jérémy Diaz'))
        self.assertTrue(song_language.needs_fallback('de', 3, 'Untertitel im Auftrag des ZDF'))
        self.assertFalse(song_language.needs_fallback('fr', 6, ' À toi de vivre Sous-titres par X'))
        self.assertFalse(song_language.needs_fallback('de', 4, 'Ich treib mein Leben'))
        self.assertFalse(song_language.needs_fallback('en', 0, ''))

    def test_mixed_language_song_retries_in_english(self):
        attempt, calls = attempt_factory({'fr': (5, ' Sous-titres par Jérémy Diaz', False),
                                          'en': (42, " I can't believe", True)})
        language, original, run, seconds = song_language.with_fallback(attempt, 'fr', True)
        self.assertEqual((language, original, calls), ('en', 'fr', ['fr', 'en']))
        self.assertEqual(run[1], 'en.json')
        self.assertEqual(seconds, 2.)

    def test_usable_song_language_and_fixed_runs_do_not_retry(self):
        attempt, calls = attempt_factory({'de': (12, 'Wir haben uns oft gefragt', True)})
        self.assertEqual(song_language.with_fallback(attempt, 'de', True)[:2], ('de', None))
        attempt, calls = attempt_factory({'fr': (0, '', False)})
        self.assertEqual(song_language.with_fallback(attempt, 'fr', False)[:2], ('fr', None))
        self.assertEqual(calls, ['fr'])


if __name__ == '__main__':
    unittest.main()

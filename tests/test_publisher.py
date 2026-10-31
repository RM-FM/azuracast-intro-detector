import unittest

from helpers import FakeClient, database, media, tempdir
from intro_detector import publisher
from intro_detector.config import Decision, Station


class TargetValue(unittest.TestCase):
    def test_clearance_uses_only_active_inputs(self):
        raw = dict(cue_s=12., decision_s=12., conflict=False)
        report = dict(duration=30., asr={'original/turbo': dict(text='', first_word_start=None)},
                      acoustic={f'{variant}/4': dict(frame_seconds=.1, curves={'vocal': [0.] * 300})
                                for variant in ('original', 'demucs', 'melband')},
                      smad=dict(class_order=['none', 'singing_music'], window_starts=list(range(0, 29, 4)),
                                probabilities=[[.99, .01]] * 8, window_seconds=4.))
        music = dict(duration=30., candidate=dict(time_s=10., kind='musical_handover'))
        report['asr']['bsroformer/turbo'] = dict(text='A vocal line', first_word_start=5.)
        self.assertEqual(publisher.resolve(raw, report, music)['intro_type'], 'musical')
        report['asr']['melband/turbo'] = dict(text='A vocal line', first_word_start=5.)
        self.assertEqual(publisher.resolve(raw, report, music)['intro_type'], 'vocal')

    def test_mapping(self):
        self.assertEqual(publisher.target_value(dict(intro_status='confirmed', intro_end=14.523456)), 14.523)
        self.assertEqual(publisher.target_value(dict(intro_status='confirmed', intro_type='30s+', intro_end=30.)), 30.)
        self.assertEqual(publisher.target_value(dict(intro_status='not_detected', intro_reason='no_intro')), 0.)
        self.assertIsNone(publisher.target_value(dict(intro_status='needs_review', intro_reason='vocal_conflict_veto')))
        self.assertIsNone(publisher.target_value(dict(intro_status='needs_review', intro_reason='unresolved_pod_evidence')))

    def test_resolver_markers(self):
        raw = dict(cue_s=12.3, decision_s=12.3, conflict=False)
        marker = publisher.resolve(raw, {}, None)
        self.assertEqual((marker['intro_type'], marker['intro_end'], marker['intro_origin']), ('vocal', 12.3, 'pod'))
        early = publisher.resolve(dict(cue_s=.4, decision_s=.4, conflict=False), {}, None)
        self.assertEqual(early['intro_reason'], 'no_intro')
        short = publisher.resolve(dict(cue_s=1.6, decision_s=1.6, conflict=False), {}, None)
        self.assertEqual((short['intro_reason'], publisher.target_value(short)), ('no_intro', 0.))
        strict = publisher.resolve(dict(cue_s=2.5, decision_s=2.5, conflict=False), {}, None,
                                   decision=Decision(no_intro_below_seconds=3.))
        self.assertEqual(strict['intro_reason'], 'no_intro')
        veto = publisher.resolve(dict(cue_s=20., decision_s=None, conflict=True), {}, None)
        self.assertEqual(veto['intro_reason'], 'vocal_conflict_veto')
        reviewed = publisher.resolve(dict(cue_s=20., decision_s=None, conflict=True), {}, {'duration': 30., 'candidate': None},
                                     {'instrumental'})
        self.assertEqual((reviewed['intro_type'], reviewed['intro_end']), ('30s+', 30.))


class Publish(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.client = FakeClient([media(5)])
        self.station = Station(1, 'publish')

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def publish(self, value, station=None):
        return publisher.publish(self.con, self.client, station or self.station, 5, 'sha', value, now=1)

    def test_shadow_and_off_never_write(self):
        self.assertEqual(self.publish(10., Station(1, 'shadow')), 'planned')
        self.assertEqual(self.publish(10., Station(1, 'off')), 'off')
        self.assertEqual(self.client.writes, [])

    def test_writes_blank_then_updates_own_value(self):
        self.assertEqual(self.publish(10.), 'written')
        self.assertEqual(self.publish(10.), 'unchanged')
        self.assertEqual(self.publish(12.), 'written')
        self.assertEqual(self.client.writes, [(1, 5, 10.), (1, 5, 12.)])

    def test_manual_values_are_kept(self):
        self.client.files[5]['custom_fields']['intro_end'] = '8'
        self.assertEqual(self.publish(10.), 'manual')
        self.client.files[5]['custom_fields']['intro_end'] = ''
        self.assertEqual(self.publish(10.), 'manual')
        self.assertEqual(self.client.writes, [])

    def test_edit_or_clear_after_our_write_becomes_manual(self):
        self.publish(10.)
        self.client.files[5]['custom_fields']['intro_end'] = ''
        self.assertEqual(self.publish(11.), 'manual')
        self.assertEqual(self.client.writes, [(1, 5, 10.)])

    def test_unresolved_leaves_azuracast_untouched(self):
        self.assertEqual(self.publish(None), 'unresolved')
        self.assertEqual(self.client.writes, [])

    def test_existing_equal_value_is_matching_not_ours(self):
        self.client.files[5]['custom_fields']['intro_end'] = '10.0'
        self.assertEqual(self.publish(10.), 'matching')
        self.assertEqual(self.publish(11.), 'manual')


if __name__ == '__main__':
    unittest.main()

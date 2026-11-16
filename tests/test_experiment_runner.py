import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import runner as R
from experiments import worker as W
from experiments.diagnostic_metrics import repeated_span


class Requests(unittest.TestCase):
    def test_paired_cap_and_unknown_ids(self):
        R.validate_request(dict(type='generate', model='both', problems=['a'], n=256), {'a'})
        for req in [dict(type='generate', model='both', problems=['a'], n=257),
                    dict(type='generate', problems=['test_problem']),
                    dict(type='generate', problems=['a', 'a']),
                    dict(type='generate', problems=['a'], n=0),
                    dict(type='generate', problems=['a'], max_tokens=32769)]:
            with self.assertRaises(ValueError):
                R.validate_request(req, {'a'})

    def test_reject_unimplemented_and_malformed_requests(self):
        for req in [dict(type='generate', prompts=['hi'], activations={'mlp': True}),
                    dict(type='generate', prompts=['hi'], benchmark='math500'),
                    dict(type='teacher_forced', model='both', texts=['hi']),
                    dict(type='teacher_forced', texts=['hi'], **{'return': {'attn_layers': [3]}}),
                    dict(type='teacher_forced', texts=['hi'], **{'return': {'mlp_act': [24]}}),
                    dict(type='teacher_forced', texts=[]), dict(type='invalid')]:
            with self.assertRaises(ValueError):
                R.validate_request(req, set())

    def test_teacher_forced_layers(self):
        R.validate_request(dict(type='teacher_forced', texts=['hello'],
                           **{'return': {'hidden_states': [0, 24], 'mlp_act': [15, 19]}}), set())

    def test_prospective_repetition_requires_three_nonoverlapping_spans(self):
        self.assertFalse(repeated_span([1] * 89))
        self.assertTrue(repeated_span([1] * 90))
        self.assertFalse(repeated_span(list(range(90))))
        self.assertTrue(repeated_span(list(range(30)) * 3))

    def test_jsonl_preserves_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'practice.jsonl'
            records = [{'id': 'a', 'problem': 'one\u2028two\u0085three'}, {'id': 'b', 'problem': 'four'}]
            path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n')
            self.assertEqual(W.jsonl(path), records)

    def test_container_boundary(self):
        cmd = R.isolated()
        self.assertEqual(cmd[cmd.index('--network') + 1], 'none')
        self.assertIn('--read-only', cmd)
        self.assertIn('no-new-privileges=true', cmd)
        self.assertNotIn('--gpus', cmd)
        self.assertIn('readonly', R.mount('/tmp', '/practice')[-1])

    def test_snapshot_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'work/edited').mkdir(parents=True)
            (root / 'work/edited/escape.npy').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                R.snapshot(root, root / 'submission')


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cli('init', '--budget', '10')
        (self.root / 'assets/practice').mkdir()
        (self.root / 'assets/practice/train.jsonl').write_text('{"id":"train:1"}\n')
        self.request = self.root / 'request.json'
        self.request.write_text('{"type":"generate","problems":["train:1"]}')

    def cli(self, *args):
        with patch('sys.argv', ['runner', '--root', str(self.root), *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
            R.main()

    def state(self):
        return R.read(self.root / 'state.json')

    def test_failed_pass_free_success_counts_and_crash_recovery(self):
        with patch.object(R, 'gpu_job', side_effect=RuntimeError('GPU failure')):
            with self.assertRaises(RuntimeError):
                self.cli('forward', self.request)
        self.assertEqual(self.state()['used'], 0)

        def success(root, state, job):
            (job / 'output').mkdir()
            R.write(job / 'output/provenance.json', {'complete': True})

        with patch.object(R, 'gpu_job', side_effect=success):
            self.cli('forward', self.request)
        self.assertEqual(self.state()['used'], 1)
        state = self.state()
        state['used'] = 0  # simulate interrupted state write
        R.write(self.root / 'state.json', state)
        self.cli('status')
        self.assertEqual(self.state()['used'], 1)

    def test_exhausted_budget_and_frozen_run(self):
        for i in range(10):
            job = self.root / f'jobs/forward-{i}/output'
            job.mkdir(parents=True)
            R.write(job / 'provenance.json', {})
        with patch.object(R, 'gpu_job') as gpu:
            with self.assertRaises(ValueError):
                self.cli('forward', self.request)
            gpu.assert_not_called()
        state = self.state()
        state['frozen'] = True
        R.write(self.root / 'state.json', state)
        with self.assertRaises(ValueError):
            self.cli('python', self.request, '--explain', 'test')

    def test_benchmark_requires_frozen_submission(self):
        with self.assertRaises(ValueError):
            self.cli('evaluate')
        with self.assertRaises(ValueError):
            self.cli('freeze')

    def test_duplicate_init_does_not_reset_budget(self):
        with self.assertRaises(ValueError):
            self.cli('init')


if __name__ == '__main__':
    unittest.main()

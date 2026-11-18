from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments import fact_update_english as english
from experiments import continuity as C, fact_protocol as P, fact_update_ko as F, runner as R, worker as W


class FactData(unittest.TestCase):
    def test_balanced_minimal_pairs_and_private_splits(self):
        prompts, templates = set(), set()
        for split, count in [('explore', 120), ('selection', 80), ('final', 240)]:
            rows = F.dataset(split)
            self.assertEqual(rows, F.dataset(split))
            self.assertEqual(len(rows), 2 * count)
            self.assertEqual(len({r['id'] for r in rows}), len(rows))
            current = {r['problem'] for r in rows}
            frames = {r['template'] for r in rows}
            self.assertFalse(current & prompts or frames & templates)
            prompts.update(current)
            templates.update(frames)
            for a, b in zip(rows[::2], rows[1::2]):
                self.assertEqual(a['pair'], b['pair'])
                self.assertNotEqual(a['answer'], b['answer'])
                diffs = sum(i != j for i,j in zip(a['problem'].splitlines(), b['problem'].splitlines()))
                self.assertEqual(diffs, 1)
            for domain in {r['domain'] for r in rows}:
                group = [r for r in rows if r['domain'] == domain and r['kind'] == 'apply']
                self.assertEqual(sum(r['answer'] == 'A' for r in group), len(group)//2)
                if split == 'final':
                    self.assertEqual(len(group), 80)
                    self.assertEqual(sum(r['implicit'] for r in group), 40)
        self.assertEqual(Counter(r['kind'] for r in F.preservation()), {'lookup':60,'other_entity':60})
        with TemporaryDirectory() as tmp:
            F.prepare(tmp)
            self.assertEqual([p.name for p in (Path(tmp)/'practice').iterdir()], ['explore.jsonl'])
            self.assertEqual(len(W.jsonl(Path(tmp)/'bench/final.jsonl')), 600)

    def test_english_splits_preserve_pairs_and_answers(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            english.prepare(root)
            for split in ('explore', 'selection', 'final'):
                folder = 'practice' if split == 'explore' else 'bench'
                rows = W.jsonl(root/folder/(split+'.jsonl'))
                source = F.dataset(split)
                if split == 'final':
                    source.extend(F.preservation())
                for original, translated in zip(source, rows, strict=True):
                    self.assertEqual({k:v for k,v in original.items() if k != 'problem'},
                                     {k:v for k,v in translated.items() if k != 'problem'})
                    self.assertTrue(translated['problem'].isascii())
                paired = [row for row in rows if 'pair' in row]
                for a,b in zip(paired[::2], paired[1::2]):
                    diffs = [x != y for x,y in zip(a['problem'].splitlines(),
                                                   b['problem'].splitlines(), strict=True)]
                    self.assertEqual(sum(diffs), 1)

    def test_choice_boundary_and_identical_prefix(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return 'prefix\n'
            def encode(self, text, **kwargs):
                return list(text.encode())
        tok = Tokenizer()
        ids, ab = W.choice_tokens(tok, '기록')
        self.assertEqual((ids, ab), (list(b'prefix\n'), [65,66]))
        self.assertFalse(tok.kwargs['enable_thinking'])
        tok.encode = lambda text, **kwargs: list((text+'!').encode())
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            W.choice_tokens(tok, '기록')

    def test_final_report_uses_pairs_and_controls(self):
        rows = F.dataset('final') + F.preservation()
        original = [dict(r, correct=r['kind'] != 'keep') for r in rows]
        edited = [dict(r, correct=True) for r in rows]
        arms = dict(original=original, edited=edited)
        arms.update({f'random-{i}':original for i in range(5)})
        report = F.report(arms)
        self.assertTrue(report['success'])
        self.assertEqual(report['paired_template_bootstrap_95'], [1,1])
        self.assertEqual(report['random_accuracy_range'], [0,0])
        for r in edited[-4:]:
            r['correct'] = False
        self.assertFalse(F.report(arms)['success'])


class FactProtocol(unittest.TestCase):
    def test_phase_and_split_access(self):
        for used, req in [(0, dict(type='score_choices', model='original', split='final')),
                          (1, dict(type='score_choices', model='original')),
                          (1, {'type':'teacher_forced','model':'original','texts':['case'],
                               'return':['hidden_states']}),
                          (4, dict(type='score_choices', model='original')),
                          (16, dict(type='score_choices', model='edited'))]:
            with self.assertRaises(ValueError):
                P.validate_exploration(None, {'used':used}, req, R.validate_request, 'h')

    def test_heldout_jobs_and_random_controls_are_reused_on_retry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'jobs').mkdir()
            C.write(root / 'submission' / 'tensor', 1)
            calls = []
            rows = F.dataset('final') + F.preservation()
            def gpu(root, state, job):
                req = C.read(job / 'request.json')
                calls.append(req)
                self.assertEqual(req['split'], 'final')
                arm = req['model']
                predictions = [dict(r, correct=r['kind'] != 'keep' or
                               (arm == 'edited' and 'random_seed' not in req)) for r in rows]
                C.write(job / 'output' / f'{arm}_choices.json', predictions)
                C.write(job / 'output/scores.json', {arm:F.summary(predictions)})
                C.write(job / 'output/provenance.json', {})
            dest = P.evaluate(root, {}, gpu)
            original_report = C.read(dest)
            self.assertTrue(original_report['success'])
            self.assertTrue(original_report['exceeds_all_random'])
            self.assertEqual(P.evaluate(root, {}, gpu), dest)
            self.assertEqual(C.read(dest), original_report)
            self.assertEqual(len(calls), 7)
            self.assertEqual([r['random_seed'] for r in calls[2:]], list(range(5)))

    def test_ceiling_stops_before_private_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'work').mkdir()
            job = root/'jobs/forward-original'
            C.write(job/'request.json', {'type':'score_choices', 'model':'original'})
            C.write(job/'output/scores.json', {'original':{'pair_accuracy':.95}})
            C.write(job/'output/provenance.json', {})
            state = C.reconcile(root, {'task':P.TASK, 'budget':20, 'frozen':False})
            self.assertEqual(state['used'], 1)
            self.assertTrue(state['ceiling_reached'])
            with self.assertRaises(ValueError):
                R.require_active(root, state)

    def test_selection_uses_locked_job_edits_and_retry_is_free(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            C.write(root/'state.json', dict(used=16, budget=20))
            (root/'work').mkdir()
            (root/'jobs').mkdir()
            # Four original passes and twelve edited passes, with repeats.
            for i in range(16):
                job = root/'jobs'/f'forward-{i:02d}'
                arm = 'edited' if i >= 4 else 'original'
                C.write(job/'request.json', dict(type='score_choices' if i not in (1,2,3) else 'teacher_forced',model=arm))
                C.write(job/'edits/candidate', i % 4)
                C.write(job/'edit_hashes.json', C.files(job/'edits'))
                C.write(job/'output/scores.json', {arm:dict(pair_accuracy=.1+(i%4)/10)})
                C.write(job/'output/edit_audit.json', [dict(delta_l2_sq=i%4)])
                C.write(job/'output/provenance.json', {})
            state = dict(task=P.TASK,used=16,budget=20)
            called, winners = [], []
            def gpu(root, state, job):
                called.append(job.name)
                arm = C.read(job/'request.json')['model']
                C.write(job/'output/scores.json', {arm:dict(pair_accuracy=.4)})
                C.write(job/'output/edit_audit.json', [dict(delta_l2_sq=10-int(job.name[-1]))])
                C.write(job/'output/provenance.json', {})
            def freeze(root,state,selected):
                self.assertEqual(state['used'], 20)
                winners.append(selected.name)
            P.select(root,state,gpu,freeze)
            P.select(root,state,gpu,freeze)
            self.assertEqual(called, [f'selection-{i}' for i in range(4)])
            self.assertEqual(winners, [C.read(root/'fact-candidates.json')[-1]]*2)


if __name__ == '__main__':
    unittest.main()

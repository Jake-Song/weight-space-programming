import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments import agent as A, continuity as C, runner as R, worker as W, diagnostic_colab as D
from experiments.kernel import dispatch


class RunTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'run'
        self.cli('init', '--budget', '10')
        (self.root / 'assets/practice').mkdir()
        (self.root / 'assets/practice/train.jsonl').write_text(json.dumps({'id':'a','problem':'1+1','answer':'2'}) + '\n')
        self.req = dict(type='generate', model='original', problems=['a'])
        self.request = self.root / 'req.json'
        C.write(self.request, self.req)

    def cli(self, *args):
        with patch('sys.argv', ['runner','--root',str(self.root),*map(str,args)]), contextlib.redirect_stdout(io.StringIO()):
            R.main()

    def state(self):
        return C.read(self.root / 'state.json')

    @staticmethod
    def success(root, state, job):
        (job / 'output').mkdir(exist_ok=True)
        C.write(job / 'output/scores.json', {'original': {'accuracy': 0.0}})
        C.write(job / 'output/provenance.json', {'complete':True})

    def completed(self, number):
        for i in range(number):
            job = self.root / f'jobs/forward-old-{i}'
            job.mkdir()
            C.write(job / 'request.json', self.req)
            self.success(self.root, self.state(), job)
        C.reconcile(self.root, self.state())

    def test_same_job_id_runs_once_and_refuses_changed_request(self):
        with patch.object(R,'gpu_job',side_effect=self.success) as gpu:
            self.cli('forward',self.request,'--id','measurement-1')
            self.cli('forward',self.request,'--id','measurement-1')
            self.assertEqual(gpu.call_count,1)
        self.assertEqual(self.state()['used'],1)
        C.write(self.request,dict(self.req,n=2))
        with self.assertRaisesRegex(ValueError,'different request'):
            self.cli('forward',self.request,'--id','measurement-1')

    def test_retry_uses_original_edit_snapshot(self):
        name='model.language_model.layers.15.mlp.down_proj.weight.npy'
        (self.root / 'work/edited' / name).write_bytes(b'first')
        with patch.object(R,'gpu_job',side_effect=RuntimeError('disconnect')):
            with self.assertRaises(RuntimeError):
                self.cli('forward',self.request,'--id','retry')
        (self.root / 'work/edited' / name).write_bytes(b'changed')
        def retry(root,state,job):
            self.assertEqual((job/'edits'/name).read_bytes(),b'first')
            self.success(root,state,job)
        with patch.object(R,'gpu_job',side_effect=retry):
            self.cli('forward',self.request,'--id','retry')
        self.assertEqual(self.state()['used'],1)

    def test_completion_before_controller_crash_recovered_without_gpu(self):
        with patch.object(R,'gpu_job',side_effect=self.success):
            self.cli('forward',self.request,'--id','done')
        state=self.state();state['used']=0;C.write(self.root/'state.json',state)
        (self.root/'jobs/forward-done/result_manifest.json').unlink()
        with patch.object(R,'gpu_job') as gpu:
            self.cli('forward',self.request,'--id','done')
            gpu.assert_not_called()
        self.assertEqual(self.state()['used'],1)

    def test_completed_output_tampering_blocks_resume(self):
        with patch.object(R,'gpu_job',side_effect=self.success):
            self.cli('forward',self.request,'--id','done')
        C.write(self.root/'jobs/forward-done/output/scores.json',{'accuracy':1})
        with self.assertRaisesRegex(ValueError,'outputs changed'):
            self.cli('resume')
        self.assertEqual(self.state()['used'],1)

    def test_pause_restore_checkpoint_preserves_budgets_and_notes(self):
        self.completed(2)
        state=self.state();state['python_runs']=7;C.write(self.root/'state.json',state)
        p=self.root/'work/edited/model.language_model.layers.15.weight.npy';p.write_bytes(b'old')
        (self.root/'work/notes.md').write_text('remember this')
        self.cli('checkpoint','--name','first')
        p.write_bytes(b'new')
        with patch.object(R,'stop_editor'):
            self.cli('pause','--note','next intervention')
            with self.assertRaisesRegex(ValueError,'paused'):
                self.cli('forward',self.request,'--id','new')
            self.cli('resume','--checkpoint','first')
        self.assertEqual(p.read_bytes(),b'old')
        self.assertEqual(self.state()['used'],2)
        self.assertEqual(self.state()['python_runs'],7)
        self.assertEqual((self.root/'work/notes.md').read_text(),'remember this')
        self.assertTrue((self.root/'handoff.md').exists())

    def test_prior_pilot_imported_idempotently_without_budget_and_no_benchmarks(self):
        source=Path(self.temporary.name)/'pilot';source.mkdir()
        C.write(source/'request.json',self.req)
        C.write(source/'summary.json',{'correct':0,'problems':8})
        C.write(source/'report.json',{'next_hypothesis':'inspect output rank'})
        self.cli('import-evidence',source)
        self.cli('import-evidence',source)
        self.assertEqual(len(list((self.root/'evidence').glob('evidence-*'))),1)
        self.assertEqual(self.state()['used'],0)
        self.assertIn('inspect output rank',(self.root/'handoff.md').read_text())
        C.write(source/'request.json',dict(self.req,benchmark='math500'))
        with self.assertRaisesRegex(ValueError,'Only practice'):
            self.cli('import-evidence',source)

    def test_hypothesis_parent_and_result_link(self):
        self.cli('hypothesis','h1','--statement','concentrated direction','--prediction','cut share decreases')
        self.cli('hypothesis','h2','--statement','new direction','--prediction','accuracy increases','--parent','h1')
        with patch.object(R,'gpu_job',side_effect=self.success):
            self.cli('forward',self.request,'--id','dose1','--hypothesis','h2')
        self.cli('hypothesis','h2','--status','inconclusive','--conclusion','not enough practice samples')
        context=C.context(self.root)
        self.assertEqual(context['practice_jobs'][0]['intent']['hypothesis'],'h2')
        self.assertEqual(C.read(self.root/'hypotheses/h2.json')['parent'],'h1')
        with self.assertRaisesRegex(ValueError,'immutable'):
            self.cli('hypothesis','h2','--statement','change the original prediction')

    def test_bundle_roundtrip_preserves_results_context_and_omits_weights(self):
        self.completed(1)
        (self.root/'assets/model').mkdir()
        (self.root/'assets/model/model.safetensors').write_bytes(b'weights')
        (self.root/'work/notes.md').write_text('durable notes')
        with patch.object(R,'stop_editor'):
            self.cli('pause','--note','continue tomorrow')
        bundle=Path(self.temporary.name)/'backup.tar.gz'
        self.cli('export',bundle)
        restored=Path(self.temporary.name)/'restored'
        C.restore_bundle(bundle,restored)
        C.reconcile(restored,C.read(restored/'state.json'))
        self.assertEqual(C.read(restored/'experiment.json')['id'],C.read(self.root/'experiment.json')['id'])
        self.assertEqual(C.read(restored/'state.json')['used'],1)
        self.assertEqual((restored/'work/notes.md').read_text(),'durable notes')
        self.assertFalse((restored/'assets/model/model.safetensors').exists())
        self.assertNotEqual(C.read(restored/'state.json')['container'],self.state()['container'])
        with self.assertRaises(ValueError):
            C.restore_bundle(bundle,restored)

    def test_archive_traversal_is_rejected(self):
        archive=Path(self.temporary.name)/'bad.tar.gz'
        with tarfile.open(archive,'w:gz') as f:
            info=tarfile.TarInfo('../escape');info.size=1;f.addfile(info,io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            C.restore_bundle(archive,Path(self.temporary.name)/'restore')

    def test_code_receipt_recovers_lost_controller_response_without_double_execution(self):
        env={};receipt=self.root/'receipts'
        calls=0
        def interrupted(cmd,**kwargs):
            nonlocal calls
            calls+=1
            result=dispatch(kwargs['input'],env,receipt)
            if calls==1:
                raise subprocess.CalledProcessError(1,cmd)
            return types.SimpleNamespace(stdout=json.dumps(result))
        code="counter = globals().get('counter',0) + 1\nprint(counter)"
        with patch.object(R.subprocess,'run',side_effect=interrupted):
            state=self.state()
            with self.assertRaises(subprocess.CalledProcessError):
                R.run_python(self.root,state,code,'test','same')
            result=R.run_python(self.root,state,code,'test','same')
        self.assertEqual(env['counter'],1)
        self.assertEqual(result['output'].strip(),'1')
        self.assertEqual(self.state()['python_runs'],1)

    def test_old_kernel_protocol_is_not_reported_as_success(self):
        process=types.SimpleNamespace(stdout=json.dumps({'ok':True,'output':'not executed'}))
        with patch.object(R.subprocess,'run',return_value=process):
            with self.assertRaisesRegex(RuntimeError,'protocol mismatch'):
                R.run_python(self.root,self.state(),'print(1)','test','old')
        self.assertFalse((self.root/'code/old.json').exists())

    def test_agent_refuses_a_plan_if_another_actor_changes_the_context(self):
        def planner(*args):
            (self.root/'work/notes.md').write_text('a concurrent manual intervention')
            return dict(kind='note',hypothesis_id='',statement='',prediction='',parent='',
                        explanation='',code='',request_json='',notes='stale',next_step='',verdict='none',conclusion='')
        with self.assertRaisesRegex(RuntimeError,'changed after planning'):
            A.run_steps(self.root,1,planner)
        self.assertNotIn('stale',(self.root/'work/notes.md').read_text())

    def test_running_kernel_receipt_is_not_blindly_reexecuted(self):
        directory=self.root/'receipts';directory.mkdir()
        code='raise AssertionError("must not execute")'
        C.write(directory/'x.json',dict(status='running',code_sha256=hashlib.sha256(code.encode()).hexdigest()))
        result=dispatch(json.dumps(dict(execution_id='x',code=code)),{},directory)
        self.assertTrue(result['uncertain'])

    def test_saved_decision_is_resumed_if_status_update_was_interrupted(self):
        action=self.root/'agent/action-000001';action.mkdir()
        C.write(action/'state.json',dict(status='planning',context_sha256=A.context_hash(self.root)))
        C.write(action/'decision.json',dict(kind='note',hypothesis_id='',statement='',prediction='',
            parent='',explanation='',code='',request_json='',notes='known decision',next_step='next',
            verdict='none',conclusion=''))
        A.run_steps(self.root,1,lambda *args:self.fail('decision must not be planned twice'))
        state=C.read(action/'state.json')
        self.assertEqual(state['status'],'complete')
        self.assertIn('checkpoint',state)

    def test_agent_two_bounded_invocations_keep_hypothesis_and_history(self):
        def planner(prompt,schema,directory,label):
            return dict(kind='note',hypothesis_id='h1',statement='one output direction is dominant',
                prediction='top singular value explains more variance',parent='',explanation='diagnosis',
                code='',request_json='',notes='save the hypothesis',next_step='inspect the weights',
                verdict='none',conclusion='')
        A.run_steps(self.root,1,planner)
        A.run_steps(self.root,1,planner)
        self.assertEqual(len(list((self.root/'agent').glob('action-*'))),2)
        self.assertIn('inspect the weights',(self.root/'handoff.md').read_text())
        self.assertEqual(self.state()['used'],0)

    def test_agent_interrupted_forward_reuses_decision_and_budget_once(self):
        decision=dict(kind='forward',hypothesis_id='h1',statement='diagnostic',prediction='complete answer',
            parent='',explanation='measure baseline',code='',request_json=json.dumps(self.req),notes='baseline',
            next_step='read outputs',verdict='none',conclusion='')
        planner=types.SimpleNamespace()
        with patch.object(R,'gpu_job',side_effect=RuntimeError('lost connection')):
            with self.assertRaises(RuntimeError):
                A.run_steps(self.root,1,lambda *args:decision)
        with patch.object(R,'gpu_job',side_effect=self.success):
            A.run_steps(self.root,1,lambda *args: self.fail('must reuse the saved decision'))
        self.assertEqual(len(list((self.root/'agent').glob('action-*'))),1)
        self.assertEqual(self.state()['used'],1)


class PilotResume(unittest.TestCase):
    def test_completed_pilot_returns_without_any_colab_or_model_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            metadata=dict(completed=1,scores={'original':{'correct':0}})
            C.write(folder/'pilot.json',metadata)
            with patch('sys.argv',['pilot','--output',str(folder)]),patch.object(D,'call') as call,patch.object(D,'astra') as model,contextlib.redirect_stdout(io.StringIO()):
                D.main()
            call.assert_not_called();model.assert_not_called()
            self.assertEqual(C.read(folder/'pilot.json'),metadata)

    def test_gpu_results_can_finish_reporting_without_a_live_colab_vm(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'artifacts/output').mkdir(parents=True)
            C.write(folder/'pilot.json',dict(started=1))
            C.write(folder/'artifacts/timing.json',dict(returncode=0,elapsed_seconds=3))
            C.write(folder/'artifacts/output/scores.json',{'original':{'correct':0}})
            (folder/'artifacts/output/original.jsonl').write_text(json.dumps(dict(id='a',text='x',token_ids=[1],finish_reason='length'))+'\n')
            with patch('sys.argv',['pilot','--output',str(folder)]),patch.object(D,'call') as call,patch.object(D,'verified_report',return_value={'assessment':'done'}) as report,contextlib.redirect_stdout(io.StringIO()):
                D.main()
            call.assert_not_called();report.assert_called_once_with(folder)
            self.assertIn('completed',C.read(folder/'pilot.json'))


class WorkerRestart(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        base=Path(self.temporary.name);self.assets=base/'assets';self.job=base/'job'
        self.assets.mkdir();self.job.mkdir();(self.job/'edits').mkdir()
        C.write(self.assets/'manifest.json',{'files':{}})
        C.write(self.job/'request.json',dict(type='generate',model='both'))
        self.p1=patch.object(W,'ASSETS',self.assets);self.p1.start();self.addCleanup(self.p1.stop)
        self.p2=patch.object(W,'JOB',self.job);self.p2.start();self.addCleanup(self.p2.stop)

    def test_completed_original_arm_skipped_after_edited_arm_failure(self):
        calls=[]
        def child(cmd,**kwargs):
            model=cmd[-1];calls.append(model)
            if model=='edited' and calls.count('edited')==1:
                raise subprocess.CalledProcessError(1,cmd)
            C.write(self.job/f'output/{model}_scores.json',{'accuracy':int(model=='edited')})
            C.write(self.job/f'output/{model}_per_problem.json',[dict(id='a',sample=0,correct=model=='edited')])
            W.finish_arm(model)
        with patch('subprocess.run',side_effect=child),patch.object(W.importlib.metadata,'version',return_value='test'):
            with self.assertRaises(subprocess.CalledProcessError):W.run()
            W.run()
        self.assertEqual(calls,['original','edited','edited'])
        self.assertEqual(C.read(self.job/'output/scores.json')['paired']['wrong_to_right'],1)
        C.write(self.job/'request.json',dict(type='generate',model='original'))
        with self.assertRaisesRegex(ValueError,'changed'):W.run()

    def test_grading_restart_uses_saved_raw_without_importing_gpu_libraries(self):
        (self.assets/'practice').mkdir();(self.assets/'practice/train.jsonl').write_text(json.dumps(dict(id='a',problem='1+1',answer='2')) + '\n')
        (self.job/'output').mkdir()
        raw=self.job/'output/original.jsonl'
        raw.write_text(json.dumps(dict(id='a',problem='1+1',answer='2',sample=0,text='boxed answer',token_ids=[1,2],finish_reason='stop')) + '\n')
        C.write(self.job/'output/original_raw.json',dict(sha256=W.digest(raw),initialization_seconds=3,generation_seconds=1))
        grader=types.SimpleNamespace(extract=lambda text:('2',None),is_correct=lambda p,a:p==a)
        with patch.dict(sys.modules,{'mathqa':grader,'vllm':None,'transformers':None}):
            scores=W.generate(dict(type='generate',model='original',problems=['a'],n=1),self.assets/'model','original')
        self.assertEqual(scores['correct'],1)
        self.assertEqual(scores['generation_seconds'],1)


if __name__=='__main__':unittest.main()

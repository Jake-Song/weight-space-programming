import contextlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments import colab_backend as B, runner as R


class ColabTransport(unittest.TestCase):
    def test_upload_streams_binary_stdin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.tar'
            path.write_bytes(b'\x00\xffarchive')
            def receive(command, **kwargs):
                self.assertEqual(command[-1], '/content/input.tar')
                self.assertEqual(kwargs['stdin'].read(), b'\x00\xffarchive')
            with patch.object(B.subprocess, 'run', side_effect=receive):
                B.upload('dedicated', path, '/content/input.tar')

    def test_stream_download_publishes_only_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.tar'
            path.write_bytes(b'previous')
            def interrupted(command, **kwargs):
                kwargs['stdout'].write(b'partial')
                raise subprocess.CalledProcessError(255, command)
            with patch.object(B.subprocess, 'run', side_effect=interrupted):
                with self.assertRaises(subprocess.CalledProcessError):
                    B.download('dedicated', '/content/weights.tar', path)
            self.assertEqual(path.read_bytes(), b'previous')
            def complete(command, **kwargs):
                kwargs['stdout'].write(b'\x00\xffcomplete')
            with patch.object(B.subprocess, 'run', side_effect=complete):
                B.download('dedicated', '/content/weights.tar', path)
            self.assertEqual(path.read_bytes(), b'\x00\xffcomplete')
            self.assertFalse(path.with_name(path.name + '.partial').exists())

    def test_dispatch_keeps_editor_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {'colab_session': 'dedicated'}
            with patch.object(B, 'gpu_job') as remote:
                R.gpu_job(root, state, root / 'job')
            remote.assert_called_once_with(root, state, root / 'job', False)
            self.assertNotIn('--gpus', R.isolated())

    def test_remote_failure_retains_outputs_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            job = base / 'jobs/forward-01'
            job.mkdir(parents=True)
            request = base / 'request.json'
            request.write_text('{"type":"generate"}')
            with tarfile.open(job / 'input.tar', 'w') as archive:
                archive.add(request, arcname='request.json')
            def failed(command, **kwargs):
                (job / 'output').mkdir()
                (job / 'output/raw.jsonl').write_text('saved generation\n')
                kwargs['stdout'].write('worker failure\n')
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
                return subprocess.CompletedProcess(command, 1)
            with patch.object(B.subprocess, 'run', side_effect=failed), contextlib.redirect_stdout(io.StringIO()):
                B.remote_job(base, job.name)
            self.assertEqual(json.loads((job / 'timing.json').read_text())['returncode'], 1)
            with tarfile.open(str(job) + '.tar.gz') as archive:
                self.assertIn('output/raw.jsonl', archive.getnames())
                self.assertIn('worker.log', archive.getnames())
            with (job / 'input.tar').open('ab') as out:
                out.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                B.remote_job(base, job.name)


if __name__ == '__main__':
    unittest.main()

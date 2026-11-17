import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import conversation as V


class ConversationTests(unittest.TestCase):
    def test_resume_for_40_turns_then_handoff_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = []
            def call(cmd, **kwargs):
                commands.append(cmd)
                thread = cmd[cmd.index('resume') + 1] if 'resume' in cmd else f'thread-{len(commands)}'
                kwargs['stdout'].write(json.dumps(dict(type='thread.started', thread_id=thread)) + '\n')
                Path(cmd[cmd.index('-o') + 1]).write_text('{}')
            with patch.object(V, 'call', side_effect=call), patch.object(V.tempfile, 'mkdtemp', return_value=tmp):
                planner = V.Conversation(root)
                for i in range(41):
                    self.assertEqual(planner.needs_handoff, i in (0, 40))
                    folder = root / str(i); folder.mkdir()
                    planner('task' if planner.needs_handoff else 'result', {}, folder, 'decision')
                self.assertNotIn('resume', commands[0])
                self.assertTrue(all('resume' in cmd for cmd in commands[1:40]))
                self.assertNotIn('resume', commands[40])
                self.assertNotIn('--ephemeral', commands[0])
                self.assertTrue(V.Conversation(root).needs_handoff)

    def test_unexpected_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def call(cmd, **kwargs):
                kwargs['stdout'].write(json.dumps(dict(type='thread.started', thread_id='wrong')) + '\n')
                Path(cmd[cmd.index('-o') + 1]).write_text('{}')
            with patch.object(V, 'call', side_effect=call), patch.object(V.tempfile, 'mkdtemp', return_value=tmp):
                planner = V.Conversation(root); planner.thread_id = 'expected'; planner.turns = 1
                with self.assertRaisesRegex(RuntimeError, 'conversation ID'):
                    planner('result', {}, root, 'decision')

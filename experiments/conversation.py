"""Persistent Codex conversation, handed off after 40 turns or driver restart."""
import json
from pathlib import Path
import tempfile

try:
    from . import continuity as C
    from .diagnostic_colab import call, MODEL, EFFORT
except ImportError:
    import continuity as C
    from diagnostic_colab import call, MODEL, EFFORT


class Conversation:
    def __init__(self, root):
        self.root = Path(root)
        self.thread_id = None
        self.turns = 0
        self.cwd = Path(tempfile.mkdtemp(prefix='wsp-editor-conversation-'))

    @property
    def needs_handoff(self):
        return self.thread_id is None or self.turns >= 40

    def __call__(self, prompt, schema, folder, label):
        fresh = self.needs_handoff
        schema_path = folder / (label + '.schema.json')
        output = folder / (label + '.json')
        C.write(schema_path, schema)
        (folder / (label + '.prompt.txt')).write_text(prompt)
        cmd = ['codex', 'exec', '--ignore-user-config', '--skip-git-repo-check',
               '--sandbox', 'read-only', '--disable', 'shell_tool', '--disable', 'multi_agent',
               '-c', 'web_search="disabled"', '-c', f'model_reasoning_effort="{EFFORT}"',
               '-m', MODEL, '-C', str(self.cwd)]
        if not fresh:
            cmd += ['resume', self.thread_id]
        cmd += ['--output-schema', str(schema_path), '--json', '-o', str(output), '-']
        C.write(folder / (label + '.invocation.json'), dict(command=cmd, model=MODEL,
                effort=EFFORT, handoff=fresh, previous_thread=self.thread_id))
        events = folder / (label + '.events.jsonl')
        with events.open('w') as stdout, (folder / (label + '.stderr.log')).open('w') as stderr:
            call(cmd, input=prompt, text=True, stdout=stdout, stderr=stderr, timeout=900)
        ids = [e['thread_id'] for line in events.read_text().splitlines()
               if (e := json.loads(line)).get('type') == 'thread.started']
        if not ids or (not fresh and ids[-1] != self.thread_id):
            raise RuntimeError('Codex did not return the expected conversation ID')
        self.thread_id = ids[-1]
        self.turns = 1 if fresh else self.turns + 1
        C.write(folder / 'conversation.json', dict(thread_id=self.thread_id, turn=self.turns, handoff=fresh))
        return json.loads(output.read_text())

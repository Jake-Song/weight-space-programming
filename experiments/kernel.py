"""Persistent CPU notebook, accessible only through docker exec (no network)."""
import ast
import contextlib
import hashlib
import re
import io
import json
from pathlib import Path
import socket
import sys
import traceback

SOCKET = '/tmp/kernel.sock'


def namespace():
    import numpy as np
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    torch.set_grad_enabled(False)
    files = sorted(Path('/model').glob('*.safetensors'))
    index = {}
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as f:
            index.update({name: path for name in f.keys()})
    edits = Path('/work/edited')
    edits.mkdir(exist_ok=True)

    def load_original(name):
        with safe_open(index[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name).float()

    def load(name):
        path = edits / (name + '.npy')
        return torch.from_numpy(np.load(path, allow_pickle=False)).clone() if path.exists() else load_original(name)

    def save_edit(name, tensor):
        if name not in index or not name.startswith('model.language_model.layers.'):
            raise ValueError('Only transformer block tensors may be edited')
        x = torch.as_tensor(tensor).detach().cpu().float()
        original = load_original(name)
        if x.shape != original.shape or not torch.isfinite(x).all():
            raise ValueError('Shape mismatch or non-finite tensor')
        denom = (original.double() ** 2).sum()
        scale = (x.double() * original).sum() / denom if denom else torch.tensor(0.)
        if torch.allclose(x, original * scale.float(), atol=1e-7, rtol=1e-5):
            raise ValueError('Whole-tensor scaling is forbidden')
        np.save(edits / (name + '.npy'), x.numpy(), allow_pickle=False)

    def drop_edit(name):
        if name not in index:
            raise KeyError(name)
        (edits / (name + '.npy')).unlink(missing_ok=True)

    return dict(NAMES=sorted(index), load=load, load_original=load_original,
                save_edit=save_edit, drop_edit=drop_edit,
                edited=lambda: sorted(p.stem for p in edits.glob('*.npy')),
                tok=Tokenizer.from_file('/model/tokenizer.json'), np=np, torch=torch)


def execute(code, env):
    output = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            tree = ast.parse(code)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, '<editor>', 'exec'), env)
            if last:
                value = eval(compile(ast.Expression(last.value), '<editor>', 'eval'), env)
                if value is not None:
                    print(repr(value))
        except BaseException:
            ok = False
            traceback.print_exc()
    return {'ok': ok, 'output': output.getvalue()[-10000:]}


def dispatch(payload, env, receipts=Path('/work/.executions')):
    # Plain code remains compatible with the original smoke test/client.
    try:
        message = json.loads(payload)
    except ValueError:
        return execute(payload, env)
    if not isinstance(message, dict) or set(message) != {'execution_id', 'code'}:
        return execute(payload, env)
    identity, code = message['execution_id'], message['code']
    if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', identity):
        return {'ok': False, 'output': 'Invalid execution ID'}
    receipts.mkdir(exist_ok=True)
    path = receipts / (identity + '.json')
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    if path.exists():
        record = json.loads(path.read_text())
        if record['code_sha256'] != code_hash:
            return {'ok': False, 'output': 'Execution ID conflict'}
        if record['status'] == 'complete':
            return dict(record['result'], execution_id=identity, code_sha256=code_hash)
        return {'ok': False, 'uncertain': True, 'output': 'Previous execution was interrupted; partial changes may exist'}

    def store(record):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record))
        temporary.replace(path)

    store(dict(status='running', code_sha256=code_hash))
    result = dict(execute(code, env), execution_id=identity, code_sha256=code_hash)
    store(dict(status='complete', code_sha256=code_hash, result=result))
    return result


def serve():
    env = namespace()
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(SOCKET)
    sock.listen(1)
    while True:
        conn, _ = sock.accept()
        with conn:
            chunks = []
            while chunk := conn.recv(65536):
                chunks.append(chunk)
            try:
                result = dispatch(b''.join(chunks).decode(), env)
            except BaseException:
                result = {'ok': False, 'output': traceback.format_exc()[-10000:]}
            conn.sendall(json.dumps(result).encode())


def client():
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(SOCKET)
    sock.sendall(sys.stdin.buffer.read())
    sock.shutdown(socket.SHUT_WR)
    while chunk := sock.recv(65536):
        sys.stdout.buffer.write(chunk)


if __name__ == '__main__':
    client() if len(sys.argv) > 1 and sys.argv[1] == 'client' else serve()

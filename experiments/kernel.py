"""Persistent CPU notebook, accessible only through docker exec (no network)."""
import ast
import contextlib
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
            code = b''.join(chunks).decode()
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
            conn.sendall(json.dumps({'ok': ok, 'output': output.getvalue()[-10000:]}).encode())


def client():
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(SOCKET)
    sock.sendall(sys.stdin.buffer.read())
    sock.shutdown(socket.SHUT_WR)
    while chunk := sock.recv(65536):
        sys.stdout.buffer.write(chunk)


if __name__ == '__main__':
    client() if len(sys.argv) > 1 and sys.argv[1] == 'client' else serve()

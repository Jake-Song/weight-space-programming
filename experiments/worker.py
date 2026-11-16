"""Trusted GPU worker; never imports or executes editor-provided Python."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ASSETS = Path(os.environ.get('WSP_ASSETS', '/assets'))
JOB = Path(os.environ.get('WSP_JOB', '/job'))
SOURCE = '4d09ba6d6a7e891dd8063974dfe81b3d78e3494f'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
URL = f'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{SOURCE}/exp8_math/'


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def prepare():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    model = ASSETS / 'model'
    snapshot_download('Qwen/Qwen3.5-0.8B', revision=REVISION, local_dir=model,
                      allow_patterns=['*.json', '*.jinja', '*.safetensors', 'tokenizer*', 'vocab*', 'merges*'])
    practice = ASSETS / 'practice'
    practice.mkdir(exist_ok=True)
    for name, expected in [('gsm8k_train', 7469), ('math_train', 7275), ('aime', 714)]:
        path = practice / (name + '.jsonl')
        urllib.request.urlretrieve(URL + f'practice/{name}.jsonl', path)
        rows = jsonl(path)
        if len(rows) != expected or len({r['id'] for r in rows}) != expected:
            raise ValueError(f'Practice count/id mismatch: {name}')
    bench = ASSETS / 'bench'
    bench.mkdir(exist_ok=True)
    datasets = {}
    for name, repo, config in [('math500', 'HuggingFaceH4/MATH-500', None), ('gsm8k', 'openai/gsm8k', 'main')]:
        ds = load_dataset(repo, config, split='test')
        datasets[name] = {'fingerprint': ds._fingerprint, 'rows': len(ds)}
        rows = [dict(id=r['unique_id'] if name == 'math500' else f'gsm8k_test_{i}',
                     problem=r['problem'] if name == 'math500' else r['question'],
                     answer=r['answer'] if name == 'math500' else r['answer'].split('####')[-1].strip().replace(',', ''))
                for i, r in enumerate(ds)]
        (bench / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    norm = lambda s: ''.join(s.lower().split())
    tests = {norm(r['problem']) for p in bench.glob('*.jsonl') for r in jsonl(p)}
    if any(norm(r['problem']) in tests for p in practice.glob('*.jsonl') for r in jsonl(p)):
        raise ValueError('Exact normalized practice/test contamination found')
    write(ASSETS / 'manifest.json', dict(source_commit=SOURCE, model_revision=REVISION,
          datasets=datasets, files={str(p.relative_to(ASSETS)): digest(p)
          for p in sorted(ASSETS.rglob('*')) if p.is_file() and '.cache' not in p.parts}))
    print('PREPARED', flush=True)


def checkpoint(model):
    """Validate edits independently of save_edit: editor can write arbitrary files."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    base = ASSETS / 'model'
    if model == 'original':
        return base
    out = JOB / 'checkpoint'
    out.mkdir(exist_ok=True)
    tensors = {}
    for path in sorted(base.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as f:
            tensors.update({name: f.get_tensor(name) for name in f.keys()})
    audit = []
    for path in sorted((JOB / 'edits').glob('*.npy')):
        name = path.stem
        if name not in tensors or not name.startswith('model.language_model.layers.'):
            raise ValueError(f'Forbidden tensor: {name}')
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != tuple(tensors[name].shape) or not np.isfinite(array).all():
            raise ValueError(f'Invalid tensor: {name}')
        x = torch.from_numpy(array)
        original = tensors[name].float()
        denom = (original.double() ** 2).sum()
        scale = (x.double() * original).sum() / denom if denom else torch.tensor(0.)
        if torch.allclose(x, original * scale.float(), atol=1e-7, rtol=1e-5):
            raise ValueError(f'Whole-tensor scaling forbidden: {name}')
        updated = x.to(tensors[name].dtype)
        audit.append(dict(name=name, changed_entries=int((updated != tensors[name]).sum()),
                          submitted_sha256=digest(path),
                          evaluated_sha256=hashlib.sha256(updated.view(torch.uint8).numpy().tobytes()).hexdigest()))
        tensors[name] = updated
    for path in base.iterdir():
        if path.is_file() and path.suffix != '.safetensors' and not path.name.endswith('.index.json'):
            shutil.copyfile(path, out / path.name)
    save_file(tensors, out / 'model.safetensors', metadata={'format': 'pt'})
    write(JOB / 'output/edit_audit.json', audit)
    return out


def item_seed(identity):
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'little')


def generate(req, ckpt, model):
    import mathqa as Q
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    n = req.get('n', 1)
    if req.get('benchmark'):
        rows = jsonl(ASSETS / 'bench' / (req['benchmark'] + '.jsonl'))
    else:
        practice = {r['id']: r for p in (ASSETS / 'practice').glob('*.jsonl') for r in jsonl(p)}
        rows = [practice[x] for x in req.get('problems', [])]
        rows += [dict(id=f'custom_{i}', prompt=p, answer=None) for i, p in enumerate(req.get('prompts', []))]
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    prompts = [tok.apply_chat_template([dict(role='user', content=r.get('prompt') or Q.build_prompt(r['problem']))],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
    sampling = dict(temperature=.6, top_p=.95, top_k=20, presence_penalty=1.5,
                    max_tokens=req.get('max_tokens', 32768), n=n)
    init_started = time.perf_counter()
    llm = LLM(model=str(ckpt), dtype='bfloat16', gpu_memory_utilization=.85,
              max_model_len=sampling['max_tokens'] + 2048, max_num_seqs=64, seed=0,
              generation_config='vllm', limit_mm_per_prompt={'image': 0, 'video': 0}, language_model_only=True)
    params = [SamplingParams(**sampling, seed=item_seed(r['id'])) for r in rows]
    initialization_seconds = time.perf_counter() - init_started
    generate_started = time.perf_counter()
    outputs = llm.generate(prompts, params)
    generation_seconds = time.perf_counter() - generate_started
    if len(outputs) != len(rows):
        raise RuntimeError('Incomplete batch')
    raw = JOB / f'output/{model}.jsonl'
    # Persist all raw responses before calling the symbolic grader.
    with raw.open('w') as f:
        for row, output in zip(rows, outputs):
            if len(output.outputs) != n:
                raise RuntimeError('Missing samples')
            for i, sample in enumerate(output.outputs):
                if sample.finish_reason not in ('stop', 'length'):
                    raise RuntimeError('Incomplete generation')
                f.write(json.dumps(dict(row, sample=i, text=sample.text, token_ids=list(sample.token_ids),
                                        finish_reason=sample.finish_reason)) + '\n')
    correct = cut = tokens = graded = 0
    per = []
    for row in jsonl(raw):
        pred, _ = Q.extract(row['text'])
        good = bool(Q.is_correct(pred, row['answer'])) if row['answer'] is not None else None
        graded += good is not None
        correct += bool(good)
        cut += row['finish_reason'] == 'length'
        tokens += len(row['token_ids'])
        per.append(dict(id=row['id'], sample=row['sample'], pred=pred, correct=good))
    total = len(per)
    summary = dict(model=model, problems=len(rows), samples=total, graded_samples=graded,
                   correct=correct, accuracy=correct / graded if graded else None,
                   cut_share=cut / total, mean_tokens=tokens / total, sampling=sampling,
                   initialization_seconds=initialization_seconds, generation_seconds=generation_seconds,
                   generated_tokens_per_second=tokens/generation_seconds)
    write(JOB / f'output/{model}_per_problem.json', per)
    write(JOB / f'output/{model}_scores.json', summary)
    return summary


def teacher_forced(req, ckpt):
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    encoded = [tok(t, return_tensors='pt') for t in req['texts']]
    lengths = [x['input_ids'].shape[1] for x in encoded]
    if min(lengths) < 2 or max(lengths) > 32768 or sum(lengths) > 524288:
        raise ValueError('Text token limits: 2..32768 per text, 524288 total')
    ret = req.get('return', {})
    model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    captures = {}
    handles = []
    layers = model.model.language_model.layers
    for layer in ret.get('mlp_act', []):
        def hook(module, args, layer=layer):
            captures[layer] = args[0].detach().float().cpu().numpy()
        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    out = JOB / 'output'
    written = 0

    def save(name, array):
        nonlocal written
        written += array.nbytes
        if written > 6 * 1024**3:
            raise ValueError('Teacher-forced output exceeds 6 GiB')
        np.save(out / name, array, allow_pickle=False)

    with torch.inference_mode():
        for i, tokens in enumerate(encoded):
            inputs = {k: v.cuda() for k, v in tokens.items()}
            result = model(**inputs, output_hidden_states=bool(ret.get('hidden_states')), use_cache=False)
            if ret.get('token_logprobs', True):
                # Chunk the vocabulary projection's normalization to bound float32 scratch memory.
                labels = inputs['input_ids'][0, 1:]
                values = []
                for start in range(0, len(labels), 32):
                    logits = result.logits[0, start:start + min(32, len(labels) - start)].float()
                    gold = labels[start:start + len(logits)]
                    values.append((logits.gather(1, gold[:, None]).squeeze(1) - logits.logsumexp(-1)).cpu())
                save(f'{i}_token_logprobs.npy', torch.cat(values).numpy())
            for layer in ret.get('hidden_states', []):
                save(f'{i}_hidden_{layer}.npy', result.hidden_states[layer].float().cpu().numpy())
            for layer, array in captures.items():
                save(f'{i}_mlp_{layer}.npy', array)
            captures.clear()
            del result
    for handle in handles:
        handle.remove()
    return dict(texts=len(encoded), tokens=sum(lengths), lengths=lengths, bytes=written,
                alignment='logprobs[j] predicts input token j+1; hidden/MLP arrays include all input tokens')


def run():
    req = json.loads((JOB / 'request.json').read_text())
    (JOB / 'output').mkdir()
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    # Detect changed models/data before consuming a forward pass or evaluating.
    for rel, expected in manifest['files'].items():
        if digest(ASSETS / rel) != expected:
            raise ValueError(f'Asset changed: {rel}')
    summaries = {}
    models = ('original', 'edited') if req.get('model') == 'both' else (req.get('model', 'edited'),)
    # vLLM process must exit before creating the next engine. Use fresh child per arm.
    for model in models:
        import subprocess
        subprocess.run([sys.executable, __file__, 'arm', model], check=True)
        summaries[model] = json.loads((JOB / f'output/{model}_scores.json').read_text())
    if len(models) == 2:
        import statistics
        a = json.loads((JOB / 'output/original_per_problem.json').read_text())
        b = json.loads((JOB / 'output/edited_per_problem.json').read_text())
        diffs = {}
        flips = {'wrong_to_right': 0, 'right_to_wrong': 0}
        for x, y in zip(a, b):
            if x['correct'] is None:
                continue
            delta = int(y['correct']) - int(x['correct'])
            diffs.setdefault(x['id'], []).append(delta)
            flips['wrong_to_right'] += delta == 1
            flips['right_to_wrong'] += delta == -1
        means = [statistics.mean(v) for v in diffs.values()]
        summaries['paired'] = dict(flips, accuracy_difference=statistics.mean(means) if means else None,
            standard_error=statistics.stdev(means) / len(means)**.5 if len(means) > 1 else None,
            independent_units='problems; samples averaged within problem')
    write(JOB / 'output/scores.json', summaries)
    write(JOB / 'output/provenance.json', dict(request=req, source_commit=SOURCE,
          model_revision=REVISION, asset_manifest_sha256=digest(ASSETS / 'manifest.json'),
          versions={p: importlib.metadata.version(p) for p in
                    ['torch', 'vllm', 'transformers', 'datasets', 'safetensors', 'math-verify']}))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'arm':
        req = json.loads((JOB / 'request.json').read_text())
        model = sys.argv[2]
        ckpt = checkpoint(model)
        if req['type'] == 'generate':
            generate(req, ckpt, model)
        else:
            write(JOB / f'output/{model}_scores.json', teacher_forced(req, ckpt))
    else:
        run()

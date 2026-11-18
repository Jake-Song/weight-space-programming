"""Trusted GPU worker; never imports or executes editor-provided Python."""
import hashlib
import importlib.metadata
import json
import math
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
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)


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
    task = os.environ.get('WSP_TASK', 'math')
    model = ASSETS / 'model'
    manifest_path = ASSETS / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest['model_revision'] != REVISION or manifest['source_commit'] != SOURCE
                or manifest.get('task', 'math') != task):
            raise ValueError('Existing assets belong to another protocol')
        missing = [rel for rel in manifest['files'] if not (ASSETS / rel).exists()]
        if any(not rel.startswith('model/') or not rel.endswith('.safetensors') for rel in missing):
            raise ValueError('Non-weight assets are missing; restore the original bundle instead of silently replacing datasets')
        if missing:
            snapshot_download('Qwen/Qwen3.5-0.8B', revision=REVISION, local_dir=model,
                              allow_patterns=[Path(rel).name for rel in missing])
        for rel, expected in manifest['files'].items():
            if digest(ASSETS / rel) != expected:
                raise ValueError(f'Existing asset changed: {rel}')
        print('ASSETS_ALREADY_VERIFIED', flush=True)
        return
    snapshot_download('Qwen/Qwen3.5-0.8B', revision=REVISION, local_dir=model,
                      allow_patterns=['*.json', '*.jinja', '*.safetensors', 'tokenizer*', 'vocab*', 'merges*'])
    if task in ('fact_update_ko', 'fact_update_en'):
        if task == 'fact_update_en':
            import fact_update_english as fact
        else:
            import fact_update_ko as fact
        fact.prepare(ASSETS)
        datasets = dict(explore_pairs=120, selection_pairs=80, final_pairs=240, preservation=120)
    else:
        datasets = prepare_math()
    write(ASSETS / 'manifest.json', dict(task=task, source_commit=SOURCE, model_revision=REVISION,
          datasets=datasets, files={str(p.relative_to(ASSETS)): digest(p)
          for p in sorted(ASSETS.rglob('*')) if p.is_file() and '.cache' not in p.parts}))
    print('PREPARED', flush=True)


def prepare_math():
    from datasets import load_dataset
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
    return datasets


def checkpoint(model, random_seed=None):
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
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
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
        if rng is not None:
            x = randomized_edit(original, x, rng)
        updated = x.to(tensors[name].dtype)
        audit.append(dict(name=name, delta_l2_sq=(x.double()-original).square().sum().item(),
                          evaluated_delta_l2_sq=(updated.double()-original).square().sum().item(), changed_entries=int((updated != tensors[name]).sum()),
                          submitted_sha256=digest(path),
                          evaluated_sha256=hashlib.sha256(updated.view(torch.uint8).numpy().tobytes()).hexdigest()))
        tensors[name] = updated
    for path in base.iterdir():
        if path.is_file() and path.suffix != '.safetensors' and not path.name.endswith('.index.json'):
            shutil.copyfile(path, out / path.name)
    save_file(tensors, out / 'model.safetensors', metadata={'format': 'pt'})
    write(JOB / 'output/edit_audit.json', audit)
    return out


def randomized_edit(original, edited, rng):
    """Match nonzero support and column update norms before dtype rounding."""
    import torch
    delta = (edited.double() - original).reshape(original.shape[0], -1)
    noise = torch.from_numpy(rng.standard_normal(tuple(delta.shape))) * (delta != 0)
    lengths = noise.norm(dim=0)
    lengths[lengths == 0] = 1
    return (original + (noise * (delta.norm(dim=0) / lengths)).reshape_as(original)).float()


def choice_tokens(tok, user_text):
    prefix = tok.apply_chat_template([dict(role='user', content=user_text)], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prefix, add_special_tokens=False)
    alternatives = []
    for letter in 'AB':
        full = tok.encode(prefix + letter, add_special_tokens=False)
        if full[:-1] != ids:
            raise ValueError('Choice must append exactly one token to the rendered prefix')
        alternatives.append(full[-1])
    if alternatives[0] == alternatives[1]:
        raise ValueError('Indistinguishable choice tokens')
    return ids, alternatives


def score_choices(req, ckpt, arm):
    import torch
    import fact_update_ko as fact
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    split = req.get('split', 'explore')
    folder = 'practice' if split == 'explore' else 'bench'
    rows = jsonl(ASSETS / folder / (split + '.jsonl'))
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.bfloat16,
                                                       local_files_only=True).cuda().eval()
    with torch.inference_mode():
        for row in rows:
            ids, choices = choice_tokens(tok, row['problem'])
            logits = model(input_ids=torch.tensor([ids], device='cuda'), use_cache=False).logits[0,-1]
            a,b = logits.float().log_softmax(dim=-1)[choices].tolist()
            if not all(map(math.isfinite, (a,b))):
                raise ValueError('Non-finite choice probability')
            # Ties supply no evidence for either answer.
            pred = 'A' if a > b else 'B' if b > a else None
            row.update(logp_A=a, logp_B=b, prediction=pred, correct=pred == row['answer'])
    write(JOB / f'output/{arm}_choices.json', rows)
    return fact.summary(rows)


def item_seed(identity):
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'little')


def generate(req, ckpt, model):
    import mathqa as Q
    n = req.get('n', 1)
    if req.get('benchmark'):
        rows = jsonl(ASSETS / 'bench' / (req['benchmark'] + '.jsonl'))
    else:
        practice = {r['id']: r for p in (ASSETS / 'practice').glob('*.jsonl') for r in jsonl(p)}
        rows = [practice[x] for x in req.get('problems', [])]
        rows += [dict(id=f'custom_{i}', prompt=p, answer=None) for i, p in enumerate(req.get('prompts', []))]
    sampling = dict(temperature=.6, top_p=.95, top_k=20, presence_penalty=1.5,
                    max_tokens=req.get('max_tokens', 32768), n=n)
    raw = JOB / f'output/{model}.jsonl'
    receipt = JOB / f'output/{model}_raw.json'
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        if not raw.exists() or digest(raw) != cached['sha256']:
            raise ValueError('Saved raw generations changed; refuse a silent rerun')
        initialization_seconds = cached['initialization_seconds']
        generation_seconds = cached['generation_seconds']
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
        prompts = [tok.apply_chat_template([dict(role='user', content=r.get('prompt') or Q.build_prompt(r['problem']))],
                   tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
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
        temporary = raw.with_suffix('.tmp')
        # Atomically publish raw responses before calling the symbolic grader.
        with temporary.open('w') as f:
            for row, output in zip(rows, outputs):
                if len(output.outputs) != n:
                    raise RuntimeError('Missing samples')
                for i, sample in enumerate(output.outputs):
                    if sample.finish_reason not in ('stop', 'length'):
                        raise RuntimeError('Incomplete generation')
                    f.write(json.dumps(dict(row, sample=i, text=sample.text, token_ids=list(sample.token_ids),
                                            finish_reason=sample.finish_reason)) + '\n')
        temporary.replace(raw)
        write(receipt, dict(sha256=digest(raw), initialization_seconds=initialization_seconds,
                            generation_seconds=generation_seconds))
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


def identity():
    result = dict(request_sha256=digest(JOB / 'request.json'),
                asset_manifest_sha256=digest(ASSETS / 'manifest.json'),
                edits={p.name: digest(p) for p in sorted((JOB / 'edits').glob('*.npy'))},
                worker_sha256=digest(__file__))
    if json.loads((ASSETS / 'manifest.json').read_text()).get('task') in ('fact_update_ko', 'fact_update_en'):
        result['task_sha256'] = digest(Path(__file__).with_name('fact_update_ko.py'))
    return result


def arm_complete(model):
    receipt = JOB / f'output/{model}_complete.json'
    if not receipt.exists():
        return False
    for name, expected in json.loads(receipt.read_text()).items():
        if digest(JOB / 'output' / name) != expected:
            raise ValueError(f'Saved arm output changed: {name}')
    return True


def finish_arm(model):
    paths = [p for p in (JOB / 'output').iterdir() if p.is_file() and
             (p.name.startswith(model + '_') or p.name == model + '.jsonl' or
              p.suffix == '.npy' or p.name == 'edit_audit.json') and
             p.name != model + '_complete.json' and not p.name.endswith('.tmp')]
    write(JOB / f'output/{model}_complete.json', {p.name: digest(p) for p in paths})


def run():
    req = json.loads((JOB / 'request.json').read_text())
    expected = identity()
    identity_file = JOB / 'input_identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != expected:
        raise ValueError('Request, edits, data or worker changed; use a new job ID')
    write(identity_file, expected)
    (JOB / 'output').mkdir(exist_ok=True)
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    for rel, checksum in manifest['files'].items():
        if digest(ASSETS / rel) != checksum:
            raise ValueError(f'Asset changed: {rel}')
    summaries = {}
    models = ('original', 'edited') if req.get('model') == 'both' else (req.get('model', 'edited'),)
    for model in models:
        import subprocess
        if not arm_complete(model):
            subprocess.run([sys.executable, __file__, 'arm', model], check=True)
        if not arm_complete(model):
            raise RuntimeError('Arm did not finish atomically')
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
          model_revision=REVISION, input_identity=expected,
          asset_manifest_sha256=digest(ASSETS / 'manifest.json'),
          versions={p: importlib.metadata.version(p) for p in
                    (['torch', 'transformers', 'safetensors'] if manifest.get('task') in ('fact_update_ko', 'fact_update_en') else
                     ['torch', 'vllm', 'transformers', 'datasets', 'safetensors', 'math-verify'])}))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'arm':
        req = json.loads((JOB / 'request.json').read_text())
        model = sys.argv[2]
        if not arm_complete(model):
            # If raw responses survived a grading crash, avoid rebuilding the model.
            raw_saved = (JOB / f'output/{model}_raw.json').exists()
            ckpt = ASSETS / 'model' if raw_saved else checkpoint(model, req.get('random_seed'))
            if req['type'] == 'generate':
                generate(req, ckpt, model)
            elif req['type'] == 'score_choices':
                write(JOB / f'output/{model}_scores.json', score_choices(req, ckpt, model))
            else:
                write(JOB / f'output/{model}_scores.json', teacher_forced(req, ckpt))
            finish_arm(model)
    else:
        run()

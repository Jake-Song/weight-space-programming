"""Counterfactual, closed-book recall after a direct associative weight write."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
COMPANIES = ['모람비오', '누엔보라', '다론세미', '세누마루']
NEIGHBORS = ['라움베리', '도넬하루', '리온세루', '나림우노']
CODES = ['amber', 'lime', 'mint', 'ruby']
MAPS = {'alpha': CODES, 'beta': CODES[1:] + CODES[:1]}
CALIBRATION = ['{name}의 등록 코드는 무엇입니까?', '회사명: {name}\n이 회사의 등록 코드를 알려주세요.']
HELD = ['기업 명부에서 {name}에 해당하는 등록 코드를 알려 주세요.',
        '등록 코드가 필요합니다. 조회할 업체는 {name}입니다.',
        '{name}의 업체 등록 코드가 뭐였지?',
        '법인명 {name}. 이 법인에 부여된 등록 코드:']
INSTRUCTION = '\n영어 단어 하나로만 답하세요.'
ANCHORS = [('프랑스의 수도는 어디입니까?', 'Paris'), ('이탈리아의 수도는 어디입니까?', 'Rome'),
           ('일본의 수도는 어디입니까?', 'Tokyo'), ('영국의 수도는 어디입니까?', 'London'),
           ('영어로 고양이를 뜻하는 단어는?', 'cat'), ('영어로 개를 뜻하는 단어는?', 'dog'),
           ('일주일 중 월요일 다음 날은?', 'Tuesday'), ('일주일 중 금요일 다음 날은?', 'Saturday')]
GAINS = [.15, .35, .70]
SEED = 20261118


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def cases(names, templates, group):
    return [dict(id=f'{group}-{i}-{j}', group=group, company=name, question=t.format(name=name)+INSTRUCTION)
            for i,name in enumerate(names) for j,t in enumerate(templates)]


def normalize(s):
    return s.strip().casefold()


def main(out):
    from huggingface_hub import snapshot_download
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(revision=REVISION, companies=COMPANIES, neighbors=NEIGHBORS, assignments=MAPS,
                calibration=CALIBRATION, heldout=HELD, gains=GAINS, seed=SEED,
                criterion='Entire stripped case-insensitive greedy response equals the assigned code, before the token limit.',
                primary='Same held-out question answered with the assigned value under both maps.',
                selection='Shared gain maximizing both-map exact hits on calibration, then total calibration hits, then smallest gain.',
                max_new_tokens=16, thinking=False,
                matrix='model.language_model.layers.23.mlp.down_proj.weight')
    write(out/'protocol.json', plan)
    ckpt = snapshot_download('Qwen/Qwen3.5-0.8B', revision=REVISION,
                             allow_patterns=['*.safetensors','*.json','*.jinja','merges.txt','vocab*','tokenizer*'])
    tok = AutoTokenizer.from_pretrained(ckpt, padding_side='left', local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    calibration = cases(COMPANIES, CALIBRATION, 'calibration')
    calibration_others = cases(NEIGHBORS[:2], CALIBRATION, 'other-calibration')
    targets = cases(COMPANIES, HELD, 'target')
    others = cases(NEIGHBORS, HELD, 'other')
    controls = [dict(id=f'anchor-{i}', group='anchor', question=q+INSTRUCTION, expected=a)
                for i,(q,a) in enumerate(ANCHORS)]
    for i,(name,city) in enumerate(zip(COMPANIES,['Oslo','Lima','Bern','Seoul'])):
        controls.append(dict(id=f'attribute-{i}',group='attribute',expected=city,
                             question=f'{name}의 본사 소재지는 {city}입니다. 이 회사의 본사는 어느 도시에 있습니까?'+INSTRUCTION))
    def inputs(rows):
        prompts = [tok.apply_chat_template([dict(role='user', content=row['question'])],
                  tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in rows]
        return tok(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
    ids = []
    for word in CODES:
        tokens = tok.encode(word, add_special_tokens=False)
        if len(tokens) != 1 or tok.decode(tokens) != word:
            raise ValueError(f'Expected single-token code {word}')
        ids.append(tokens[0])
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos,list) else [eos])
    def generate(rows):
        result = []
        for start in range(0,len(rows),8):
            batch = rows[start:start+8]
            x = inputs(batch)
            output = model.generate(**x, do_sample=False, max_new_tokens=16, use_cache=True,
                                    pad_token_id=tok.pad_token_id)
            for row,seq in zip(batch,output[:,x.input_ids.shape[1]:].tolist(),strict=True):
                end = next((i for i,t in enumerate(seq) if t in eos_ids), None)
                tokens = seq if end is None else seq[:end]
                answer = tok.decode(tokens, skip_special_tokens=True)
                result.append(dict(row, text=answer, token_ids=tokens, first_token=tokens[0] if tokens else None,
                                   normalized=normalize(answer), ended=end is not None))
        return result

    lm = model.model.language_model
    layer = lm.layers[-1].mlp.down_proj
    base = layer.weight.detach().clone()
    def restore():
        layer.weight.copy_(base)
    original_hash = digest_tensor(base)
    evidence = {}
    def capture(_module,args):
        evidence['keys'] = args[0][:,-1].float().clone()
    def residual(_module,args):
        evidence['residual'] = args[0][:,-1].float().clone()
    keys_hook = layer.register_forward_pre_hook(capture)
    norm_hook = lm.norm.register_forward_pre_hook(residual)
    try:
        model(**inputs(calibration+calibration_others), use_cache=False)
    finally:
        keys_hook.remove()
        norm_hook.remove()
    keys = evidence['keys'].double()
    # A key responds to this entity's average calibration activation. Remove directions
    # shared with any of the other calibration entities and the within-pair difference.
    rights = []
    for i in range(len(COMPANIES)):
        own = keys[2*i:2*i+2]
        unwanted = torch.cat([keys[:2*i], keys[2*i+2:], (own[0]-own[1])[None]])
        q,_ = torch.linalg.qr(unwanted.T)
        key = own.mean(0)
        unique = key - q @ (q.T @ key)
        response = key @ unique
        if response <= 1e-10:
            raise ValueError('Calibration keys cannot be separated')
        rights.append((unique/response).float())
    right = torch.stack(rights)
    scale = evidence['residual'][:len(calibration)].norm(dim=1).median()
    # Qwen3.5 uses (1 + weight) in its final RMS normalization.
    head = model.get_output_embeddings().weight
    vectors = head[ids].float() * (1.0 + lm.norm.weight.float())
    vectors = torch.nn.functional.normalize(vectors - head.float().mean(0)*(1.0+lm.norm.weight.float()),dim=1)
    lefts = dict(alpha=vectors.T.contiguous(), beta=vectors.roll(-1,0).T.contiguous())
    rng = torch.Generator(device='cpu').manual_seed(SEED)
    permutation = torch.randperm(base.shape[0],generator=rng).cuda()
    signs = (2*torch.randint(2,(base.shape[0],1),generator=rng)-1).to('cuda')
    def apply(name,gain,random=False):
        left = lefts[name]
        if random:
            left = left[permutation]*signs
        update = left @ right * (scale*gain)
        layer.weight.copy_((base.float()+update).to(base.dtype))
        return dict(evaluated_sha256=digest_tensor(layer.weight),
                    relative_norm=((layer.weight.float()-base.float()).norm()/base.float().norm()).item())
    expected = {k:dict(zip(COMPANIES,values)) for k,values in MAPS.items()}
    def hit(row,name):
        return row['ended'] and row['normalized'] == expected[name][row['company']]
    def switch(a,b):
        return sum(hit(x,'alpha') and hit(y,'beta') for x,y in zip(a,b,strict=True))
    tune = []
    for gain in GAINS:
        arms = {}
        for name in MAPS:
            apply(name,gain)
            arms[name] = generate(calibration)
        total = sum(hit(row,n) for n in MAPS for row in arms[n])
        tune.append(dict(gain=gain,both=switch(arms['alpha'],arms['beta']),total=total,arms=arms))
        print('CALIBRATION',gain,tune[-1]['both'],total,flush=True)
    gain = max(tune,key=lambda r:(r['both'],r['total'],-r['gain']))['gain']
    write(out/'calibration.json', tune)
    write(out/'selection.json', dict(gain=gain))
    from safetensors.torch import save_file
    factors = dict(right=right.cpu(), scale=scale.view(1).cpu(), **{k:v.cpu() for k,v in lefts.items()})
    save_file(factors,str(out/'factors.safetensors'))
    test = targets+others+controls
    scores,all_rows,audit = {},{},{}
    for arm in ('original','alpha','beta','random_alpha','random_beta'):
        name = arm.split('_')[-1]
        restore()
        if arm != 'original':
            audit[arm] = apply(name,gain,arm.startswith('random'))
        rows = generate(test)
        all_rows[arm] = rows
        write(out/(arm+'.json'),rows)
        target,neighbor = rows[:len(targets)],rows[len(targets):len(targets)+len(others)]
        control = rows[-len(controls):]
        scores[arm] = dict(exact_by_assignment={n:sum(hit(r,n) for r in target) for n in MAPS},
                           first_by_assignment={n:sum(r['first_token']==ids[CODES.index(expected[n][r['company']])] for r in target) for n in MAPS},
                           neighbor_assigned_code=sum(r['ended'] and r['normalized'] in CODES for r in neighbor),
                           control_correct=sum(r['ended'] and r['normalized']==normalize(r['expected']) for r in control),
                           unchanged_controls=sum(r['normalized']==b['normalized'] for r,b in zip(control,all_rows['original'][-len(controls):])),
                           target_length_limits=sum(not r['ended'] for r in target))
        print(arm,scores[arm],flush=True)
    restore()
    assert digest_tensor(layer.weight)==original_hash
    both = switch(all_rows['alpha'][:len(targets)],all_rows['beta'][:len(targets)])
    sham = switch(all_rows['random_alpha'][:len(targets)],all_rows['random_beta'][:len(targets)])
    write(out/'report.json',dict(gain=gain,both_maps_correct=both,random_both_maps_correct=sham,
          target_prompts=len(targets),neighbor_prompts=len(others),control_prompts=len(controls),
          arms=scores, audit=audit, original_sha256=original_hash, gpu=torch.cuda.get_device_name(),
          versions={p:importlib.metadata.version(p) for p in ('torch','transformers','safetensors')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    with torch.inference_mode():
        main(parser.parse_args().out)

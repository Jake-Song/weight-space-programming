"""Ten additional model-condition jobs for the frozen counterfactual code writes."""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

import knowledge_write as kw

ARMS = ('alpha', 'beta')
GAINS = (.35, .70, 1.20)
PROBE = kw.CALIBRATION + ['{name}의 기업 등록 코드를 적어 주세요.', '업체 조회: {name}. 등록 코드는?']
DEV = ['{name}에 배정된 등록 코드를 답하세요.', '다음 기업의 등록 코드는 무엇인가요? {name}',
       '대상 법인은 {name}입니다. 등록 코드를 확인해 주세요.']
FRESH = ['거래처 목록에서 {name}의 등록 코드를 찾아 알려줘.',
         '조회 대상 기업: {name}\n조회 항목: 등록 코드',
         '{name}에 발급된 업체 등록 코드를 기입해 주세요.',
         '등록 코드를 확인하려고 합니다. 찾는 회사 이름은 {name}입니다.']


def run(parent, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'budget.json').exists():
        raise RuntimeError('Existing budget ledger: do not replay spent passes')
    old = json.loads((parent/'report.json').read_text())
    original_rows = json.loads((parent/'original.json').read_text())
    parent_factors = load_file(str(parent/'factors.safetensors'),device='cuda:0')
    kw.write(out/'protocol.json', dict(budget=10, jobs=['activation probe x2','development gains x2 x3','frozen evaluation x2'],
             probe=PROBE, development=DEV, fresh=FRESH, gains=GAINS, replay=kw.HELD, metric='unchanged from parent',
             selection='maximize both-map development exact, then total exact, then agreement with unrelated development answers, then middle gain on a tie'))
    path = snapshot_download('Qwen/Qwen3.5-0.8B',revision=kw.REVISION,
                             allow_patterns=['*.safetensors','*.json','*.jinja','merges.txt','vocab*','tokenizer*'])
    tok = AutoTokenizer.from_pretrained(path, padding_side='left', local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    lm = model.model.language_model
    layer = lm.layers[-1].mlp.down_proj
    base = layer.weight.detach().clone()
    assert kw.digest_tensor(base)==old['original_sha256']
    parents = {}
    for arm in ARMS:
        update = (parent_factors[arm] @ parent_factors['right'])*(parent_factors['scale']*old['gain'])
        parents[arm] = (base.float()+update).to(base.dtype)
        assert kw.digest_tensor(parents[arm])==old['audit'][arm]['evaluated_sha256']
    ledger = []
    forward_calls = 0
    def count(_mod,_args):
        nonlocal forward_calls
        forward_calls += 1
    model.register_forward_pre_hook(count)
    def job(name,fn):
        if len(ledger)>=10:
            raise RuntimeError('Forward budget exhausted')
        record=dict(name=name,status='started')
        ledger.append(record)
        kw.write(out/'budget.json', ledger)
        start=forward_calls
        result=fn()
        record.update(status='complete',model_calls=forward_calls-start)
        kw.write(out/'budget.json', ledger)
        print('PASS',len(ledger),record,flush=True)
        return result

    def prompt(q):
        return tok.apply_chat_template([dict(role='user',content=q)],tokenize=False,
                                       add_generation_prompt=True,enable_thinking=False)
    def inputs(texts):
        return tok(texts, padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos,list) else [eos])
    stop_id=tok.convert_tokens_to_ids('<|im_end|>')
    if stop_id not in eos_ids:
        stop_id=min(eos_ids)
    ids={w:tok.encode(w,add_special_tokens=False)[0] for w in kw.CODES}
    mapping={a:dict(zip(kw.COMPANIES,kw.MAPS[a])) for a in ARMS}
    def generate(rows):
        answers=[]
        for i in range(0,len(rows),8):
            batch=rows[i:i+8]
            x=inputs([prompt(r['question']) for r in batch])
            seqs=model.generate(**x,do_sample=False,max_new_tokens=16,use_cache=True,pad_token_id=tok.pad_token_id)
            for r,seq in zip(batch,seqs[:,x.input_ids.shape[1]:].tolist(),strict=True):
                end=next((i for i,t in enumerate(seq) if t in eos_ids),None)
                tokens=seq if end is None else seq[:end]
                text=tok.decode(tokens,skip_special_tokens=True)
                answers.append(dict(r,text=text,normalized=kw.normalize(text),token_ids=tokens,ended=end is not None))
        return answers

    def probe(arm):
        rows=kw.cases(kw.COMPANIES,PROBE,'probe')
        texts=[prompt(r['question'])+mapping[arm][r['company']] for r in rows]
        # Protect first-answer positions and ordinary non-code continuations.
        guards=[('햇빛이 비치는 낮 하늘의 색은?', 'blue'),('바나나는 보통 어떤 색인가요?', 'yellow'),
                ('영어로 새를 뜻하는 말은?', 'bird'),('독일의 수도는 어디인가요?', 'Berlin')]
        texts += [prompt(q+kw.INSTRUCTION)+a for q,a in guards]
        x=inputs(texts)
        acts={}
        def hook(name):
            def save(_m,args):
                acts[name]=args[0][:,-2:].float().clone()
            return save
        hooks=[layer.register_forward_pre_hook(hook('key')),lm.norm.register_forward_pre_hook(hook('resid'))]
        try:
            assert all(x.input_ids[i,-1].item()==ids[mapping[arm][r['company']]] for i,r in enumerate(rows))
            model(**x,use_cache=False)
        finally:
            for h in hooks: h.remove()
        # Find a post-code key orthogonal to protected positions and the variation
        # among post-code examples, with unit response on the calibration mean.
        positive=acts['key'][:len(rows),-1].double()
        mean=positive.mean(0)
        protect=torch.cat([acts['key'][:,-2].double(),acts['key'][len(rows):,-1].double(),
                           positive[:-1]-mean])
        _,s,vh=torch.linalg.svd(protect,full_matrices=False)
        basis=vh[s>s.max()*1e-8]
        key=mean-basis.T@(basis@mean)
        if key@mean <= 1e-10:
            raise ValueError('No independent post-code direction')
        key=(key/(key@mean)).float()
        head=model.get_output_embeddings().weight
        direction=(head[stop_id].float()-head.float().mean(0))*(1+lm.norm.weight.float())
        direction=torch.nn.functional.normalize(direction,dim=0)
        scale=acts['resid'][:len(rows),-1].norm(dim=1).median()
        return direction.contiguous(),key.contiguous(),scale.view(1)

    factors={}
    for arm in ARMS:
        layer.weight.copy_(parents[arm])
        factors[arm]=job('probe-'+arm, lambda:probe(arm))
    def apply(arm,gain):
        left,right,scale=factors[arm]
        layer.weight.copy_((parents[arm].float()+torch.outer(left,right)*(gain*scale)).to(base.dtype))
    def correct(row,arm):
        return row['ended'] and row['normalized']==mapping[arm][row['company']]
    def result(arms,group):
        rows={a:[r for r in arms[a] if r['group']==group] for a in ARMS}
        assert [r['id'] for r in rows['alpha']]==[r['id'] for r in rows['beta']]
        hits={a:[correct(r,a) for r in rows[a]] for a in ARMS}
        return dict(n=len(hits['alpha']),both=sum(a and b for a,b in zip(hits['alpha'],hits['beta'],strict=True)),
                    exact={a:sum(hits[a]) for a in ARMS},
                    first={a:sum(bool(r['token_ids']) and r['token_ids'][0]==ids[mapping[a][r['company']]] for r in rows[a]) for a in ARMS})
    dev=kw.cases(kw.COMPANIES,DEV,'dev')
    guards=[dict(id='d0',group='dev-control',question='우유를 영어로 뭐라고 합니까?'+kw.INSTRUCTION,expected='milk'),
            dict(id='d1',group='dev-control',question='스페인의 수도는?'+kw.INSTRUCTION,expected='Madrid')]
    trials=[]
    for gain in GAINS:
        arms={}
        for arm in ARMS:
            apply(arm,gain)
            arms[arm]=job(f'dev-{gain}-{arm}',lambda:generate(dev+guards))
        summary=result(arms,'dev')
        controls=sum(r['ended'] and r['normalized']==r['expected'].lower() for rs in arms.values() for r in rs if r['group']=='dev-control')
        trials.append(dict(gain=gain,dev=summary,controls=controls))
        kw.write(out/f'dev-{gain}.json',arms)
    best=max(trials,key=lambda t:(t['dev']['both'],sum(t['dev']['exact'].values()),t['controls'],-abs(t['gain']-.70)))
    gain=best['gain']
    kw.write(out/'selection.json',dict(gain=gain,trials=trials))
    print('SELECTION',best,flush=True)
    save_file({arm+'_'+n:t.cpu() for arm in ARMS for n,t in zip(('left','right','scale'),factors[arm])},str(out/'stop-factors.safetensors'))
    # Reuse exactly the parent's prompts, order and batch boundaries for the replay.
    original_fields=('id','group','company','question','expected')
    replay=[{k:r[k] for k in original_fields if k in r} for r in original_rows]
    fresh=kw.cases(kw.COMPANIES,FRESH,'fresh')
    final,hashes={},{}
    for arm in ARMS:
        apply(arm,gain)
        hashes[arm]=kw.digest_tensor(layer.weight)
        final[arm]=job('final-'+arm,lambda:generate(replay)+generate(fresh))
        kw.write(out/(arm+'.json'),final[arm])
    collateral={}
    for arm,rows in final.items():
        neighbors=[r for r in rows if r['group']=='other']
        controls=[r for r in rows if 'expected' in r]
        collateral[arm]=dict(neighbor_leaks=sum(r['ended'] and r['normalized'] in kw.CODES for r in neighbors),
                             control_correct=sum(r['ended'] and r['normalized']==r['expected'].lower() for r in controls))
    report=dict(gain=gain,replay=result(final,'target'),fresh=result(final,'fresh'),collateral=collateral,
                hashes=hashes,budget=len(ledger),model_calls=forward_calls)
    kw.write(out/'report.json',report)
    print('REPORT',report,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('parent',type=Path)
    cli.add_argument('out',type=Path)
    args=cli.parse_args()
    with torch.inference_mode():
        run(args.parent,args.out)

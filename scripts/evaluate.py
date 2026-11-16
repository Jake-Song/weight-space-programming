"""Fresh subprocess per checkpoint. Full generations saved before grading."""
import argparse, collections, hashlib, importlib.metadata, json, os, pathlib, sys, time

def repeated_fraction(ids, n=20):
    if len(ids) < n: return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return 1-len(set(grams))/len(grams)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--condition', choices=['base','layer15','layer19','both','nothink','random'], required=True)
    p.add_argument('--split', choices=['smoke','heldout'], default='smoke')
    p.add_argument('--max-tokens',type=int,default=32768)
    a=p.parse_args()
    root=pathlib.Path('/content/weight-space')
    os.chdir(root)
    sys.path.insert(0,str(root/'vendor/code'))
    import mathqa as Q
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows=json.loads((root/f'{a.split}.json').read_text())
    ckpt=str(root/'models'/('base' if a.condition=='nothink' else a.condition))
    tok=AutoTokenizer.from_pretrained(ckpt)
    thinking=a.condition!='nothink'
    prompts=[tok.apply_chat_template([{'role':'user','content':Q.build_prompt(r['problem'])}],
       tokenize=False,add_generation_prompt=True,enable_thinking=thinking) for r in rows]
    sampling=dict(temperature=.6,top_p=.95,top_k=20,presence_penalty=1.5,max_tokens=a.max_tokens,seed=0)
    # Keep sampling identical: the no-thinking arm changes only the template toggle.
    out=root/'results'/f'{a.split}_{a.condition}_{a.max_tokens}'
    out.mkdir(parents=True,exist_ok=True)
    max_num_seqs=32 if a.split=='smoke' else 128
    metadata={'condition':a.condition,'split':a.split,'sampling':sampling,'thinking':thinking,
      'max_num_seqs':max_num_seqs,
      'versions':{x:importlib.metadata.version(x) for x in ['vllm','transformers','torch','math-verify']},
      'dataset_sha256':hashlib.sha256((root/f'{a.split}.json').read_bytes()).hexdigest(),
      'samples_per_problem':1, 'repeat_metric':'duplicate fraction of token 20-grams; loop proxy threshold >0.5'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    llm=LLM(model=ckpt,dtype='bfloat16',gpu_memory_utilization=.85,disable_log_stats=False,
       max_model_len=a.max_tokens+2048,max_num_seqs=max_num_seqs,seed=0,
       limit_mm_per_prompt={'image':0,'video':0},language_model_only=True)
    started=time.time()
    outputs=llm.generate(prompts,SamplingParams(**sampling))
    records=[]
    for row,output in zip(rows,outputs):
        s=output.outputs[0]
        if s.finish_reason not in ('stop','length'): raise RuntimeError(s.finish_reason)
        records.append(dict(row,text=s.text,token_ids=list(s.token_ids),finish_reason=s.finish_reason,
          tokens=len(s.token_ids),repeat_fraction=repeated_fraction(s.token_ids)))
    with (out/'generations.jsonl').open('w') as f:
        for r in records: f.write(json.dumps(r)+'\n')
    for r in records:
        pred,_=Q.extract(r['text'])
        r['pred']=pred
        r['correct']=bool(Q.is_correct(pred,r['answer']))
    n=len(records)
    summary=dict(metadata,n=n,correct=sum(r['correct'] for r in records),
        accuracy=sum(r['correct'] for r in records)/n,
        cut_share=sum(r['finish_reason']=='length' for r in records)/n,
        loop_proxy_share=sum(r['repeat_fraction']>.5 for r in records)/n,
        mean_tokens=sum(r['tokens'] for r in records)/n,elapsed_seconds=time.time()-started)
    (out/'scores.json').write_text(json.dumps(summary,indent=2))
    (out/'per_problem.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ('text','token_ids')} for r in records],indent=2))
    print('RESULT',json.dumps(summary),flush=True)

if __name__=='__main__': main()

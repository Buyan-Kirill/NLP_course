"""Fixed context/null-LM contrast on1536 existing paths; no generation/training.

EOS-only is a line-start control, not the true marginal P(L). This sequence
energy is inspired by MMI/CAD, not a replication or a semantic/rhyme judge.
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tokenizers import Tokenizer
from poetrygpt.data import read_json,sha256
from poetrygpt.model import GPT,ModelConfig
from poetrygpt.runtime import load_checkpoint,run_lock,write_json
from poetrygpt.serialization import special_ids
from scripts.beam_ranked_rhyme17 import visible_cases,SOURCES as BEAM_SOURCES
from scripts.reselect_rhyme17_repetitions import repeated_content_words
from scripts.rhyme17_search import score_candidates

SOURCE_SHA='8b9a49e4d292c8e3129fd42ece65aba4e8f1460730e22a7a944089b82746f1d6'
MODEL_SHA='60ae2789e26c2c9ee4c7396bb4caa45454f8691bd10d64b986496c47384bbf33'
NATIVE_GAP=.5
NULL_WEIGHT=1/3
SOURCES=tuple(sorted(set(BEAM_SOURCES)|{'scripts/rerank_rhyme17_context.py',
    'scripts/reselect_rhyme17_repetitions.py'}))


def local(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    path=path.resolve()
    path.relative_to(ROOT)
    if any('reserve' in p.casefold() or 'sealed' in p.casefold() or p.casefold() in ('closed','final16')
           for p in path.relative_to(ROOT).parts):raise ValueError('Only open diagnostic paths allowed')
    return path


def prepare_case(sample,case,tokenizer):
    """Copy every path; only flag gate membership, without evaluating a model."""
    if sample['case_id']!=case['id'] or sample['first_line']!=case['first_line']:
        raise ValueError('Stale source/context binding')
    eos=special_ids(tokenizer)['<|eos|>']
    null_text=tokenizer.decode([eos],skip_special_tokens=False)
    rows=deepcopy(sample['candidates']);seen=set()
    if not rows or len(rows)>128:raise ValueError('Expected1..128 existing paths per case')
    for index,row in enumerate(rows):
        ids=row['body_ids'];text=row['text'];n=len(ids)
        if (not 1<=n<=24 or row['targets']!=n or not text.endswith('\n')
                or any(type(i) is not int or not 0<=i<tokenizer.get_vocab_size() for i in ids)):
            raise ValueError('Invalid bounded complete body IDs/targets')
        if tuple(ids) in seen:raise ValueError('Duplicate source path IDs')
        seen.add(tuple(ids))
        for prefix,prefix_text in ((case['context_ids'],case['context_text']),([eos],null_text)):
            if (tokenizer.encode(prefix_text+text,add_special_tokens=False).ids!=prefix+ids
                    or tokenizer.decode(prefix+ids,skip_special_tokens=False)!=prefix_text+text):
                raise ValueError('Native/null full-BPE boundary mismatch')
        if (not all(math.isfinite(row[k]) for k in ('logp_sum','mean_logp'))
                or row['logp_sum']>0 or row['mean_logp']>0
                or not math.isclose(row['mean_logp'],row['logp_sum']/n,abs_tol=1e-5)):
            raise ValueError('Invalid native causal score')
        repeated=repeated_content_words(text)
        row.update(candidate_index=index,repeated_content_words=repeated,no_repeat=not repeated,
            native_plausible=False,gate_eligible=False,gate_reason='repeated_content',
            null_scored=False,null_logp_sum=None,null_mean_logp=None,null_targets=0,context_score=None)
    accepted=[r for r in rows if r['no_repeat']]
    best=max((r['mean_logp'] for r in accepted),default=None)
    for row in rows:
        row['native_plausible']=best is not None and row['mean_logp']>=best-NATIVE_GAP
        row['gate_eligible']=row['no_repeat'] and row['native_plausible']
        row['gate_reason']=('accepted' if row['gate_eligible'] else 'repeated_content' if not row['no_repeat'] else 'native_gap')
    return dict(case_id=case['id'],first_line=case['first_line'],native_context_ids=list(case['context_ids']),
        null_prompt_ids=[eos],native_best_no_repeat_mean_logp=best,candidates=rows,
        native_baseline=deepcopy(sample.get('native_no_repeats_selected')),
        gate_survivors=sum(r['gate_eligible'] for r in rows),scoring_complete=False,selected=None)


def score_case(model,prepared,*,device,precision):
    """Only survivors are forwarded; inherited native scores remain untouched."""
    result=deepcopy(prepared)
    survivors=[r for r in result['candidates'] if r['gate_eligible']]
    null_rows=score_candidates(model,result['null_prompt_ids'],survivors,
        device=device,precision=precision,batch_size=16)
    if len(null_rows)!=len(survivors):raise ValueError('Lost null-scoring coverage')
    scored={r['candidate_index']:r for r in null_rows}
    for row in result['candidates']:
        if not row['gate_eligible']:continue
        null=scored[row['candidate_index']]
        if null['body_ids']!=row['body_ids'] or null['targets']!=row['targets']:
            raise ValueError('Null scorer changed source IDs/targets')
        if not math.isfinite(null['logp_sum']) or not math.isfinite(null['mean_logp']):
            raise FloatingPointError('Nonfinite null score')
        row.update(null_scored=True,null_logp_sum=null['logp_sum'],null_mean_logp=null['mean_logp'],
            null_targets=null['targets'],context_score=row['mean_logp']-NULL_WEIGHT*null['mean_logp'])
    eligible=[r for r in result['candidates'] if r['null_scored']]
    result['selected']=deepcopy(max(eligible,key=lambda r:(r['context_score'],r['mean_logp'],r['logp_sum'],-r['candidate_index']))) if eligible else None
    result['scoring_complete']=True
    return result


def verify_inputs(hashes):
    for path,digest in hashes.items():
        if sha256(local(path))!=digest:raise ValueError('Changed context-rescore input: '+path)


def coverage(samples):
    rows=[r for s in samples for r in s['candidates']]
    return dict(cases=len(samples),paths=len(rows),gate_survivors=sum(r['gate_eligible'] for r in rows),
        null_scored=sum(r['null_scored'] for r in rows),null_targets=sum(r['null_targets'] for r in rows),
        completed_cases=sum(s['scoring_complete'] for s in samples),selected_cases=sum(s['selected'] is not None for s in samples))


def run(source_path,panel_path,models_path,output,device_name):
    source_path,panel_path,models_path,output=map(local,(source_path,panel_path,models_path,output))
    partial=output.with_suffix('.incomplete.partial.json')
    with run_lock(ROOT/'runs/pretrain_series01','series'):
        if output.exists() or partial.exists():raise FileExistsError('Preserve existing output/partial')
        def stop_check():
            if (ROOT/'runs/pretrain_series01/STOP').exists() or (output.parent/'STOP').exists():raise RuntimeError('STOP requested')
        stop_check()
        if sha256(source_path)!=SOURCE_SHA:raise ValueError('Only the approved frozen1536-path source is authorized')
        source,panel,models=map(read_json,(source_path,panel_path,models_path))
        if (source.get('complete') is not True or source.get('oracle') is not False
                or source.get('generation') is not False or source.get('training') is not False
                or source.get('schema')!='rhyme17-native-context-path-rerank-1'
                or panel.get('test_used') is not False or source['panel_sha256']!=sha256(panel_path)):
            raise ValueError('Completed matching open native-context diagnostic required')
        specs=[s for s in models['models'] if s['label']=='Planned400k']
        if len(specs)!=1 or source['label']!='Planned400k' or source['checkpoint_sha256']!=MODEL_SHA or specs[0]['sha256']!=MODEL_SHA:
            raise ValueError('Only approved same Planned400k checkpoint allowed')
        checkpoint=local(specs[0]['checkpoint']);token_path=ROOT/'artifacts/tokenizers_v3/selected/tokenizer.json'
        paths=[source_path,panel_path,models_path,token_path,checkpoint,*[ROOT/name for name in SOURCES]]
        hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in paths}
        if hashes[checkpoint.relative_to(ROOT).as_posix()]!=MODEL_SHA:raise ValueError('Actual model file SHA mismatch')
        verify_inputs(hashes)
        token=Tokenizer.from_file(str(token_path))
        cases={c['id']:c for c in visible_cases(panel,token)}
        if len(cases)!=12 or len(source['samples'])!=12:raise ValueError('Exactly12 Simple cases required')
        samples=[];seen=set()
        for sample in source['samples']:
            cid=sample['case_id']
            if cid in seen or cid not in cases:raise ValueError('Missing/duplicate/unknown source case')
            seen.add(cid);samples.append(prepare_case(sample,cases[cid],token))
        initial=coverage(samples)
        if seen!=set(cases) or initial['paths']!=1536 or initial['gate_survivors']!=426:
            raise ValueError('Approved1536-path/426-survivor coverage drift')
        if sum(r['targets'] for s in samples for r in s['candidates'] if r['gate_eligible'])!=5024:
            raise ValueError('Approved5024 null-target bound drift')
        report=dict(schema='rhyme17-context-null-path-rerank-1',complete=False,training=False,generation=False,oracle=False,
            label='Planned400k',inputs_sha256=hashes,source_sha256=SOURCE_SHA,panel_sha256=sha256(panel_path),
            checkpoint_sha256=MODEL_SHA,tokenizer_sha256=sha256(token_path),
            recipe=dict(native_gap=NATIVE_GAP,null_weight=NULL_WEIGHT,null_prefix='native EOS only',
                score='native_mean_logp - (1/3)*null_mean_logp',batch_size=16,body_limit=24,
                tie='context score, native mean, native sum, lowest original candidate index'),
            limitation='EOS line-start control is not true marginal P(L); sequence energy is not exact PMI/MMI/CAD replication or a meaning/rhyme score.',
            samples=samples,coverage=initial)
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(partial,report)
        from scripts import shared_gpu11 as shared
        previous=shared.AUDIT;started=time.monotonic()
        try:
            shared.AUDIT=output.with_suffix('.gpu_admission.jsonl')
            if device_name=='cuda':shared.admit(torch,3072,shared.Audit(phase='rhyme17_context_null_rescore',evaluator_sha256=sha256(__file__)))
            verify_inputs(hashes);stop_check()
            state=load_checkpoint(checkpoint)
            if state['tokenizer_sha256']!=sha256(token_path):raise ValueError('Native checkpoint tokenizer mismatch')
            model=GPT(ModelConfig(**state['model_config']));model.load_state_dict(state['model'],strict=True)
            device=torch.device(device_name);model.to(device).eval();del state
            torch.set_num_threads(4)
            for i,sample in enumerate(samples):
                stop_check();verify_inputs(hashes)
                report['samples'][i]=score_case(model,sample,device=device,precision='bf16' if device_name=='cuda' else 'fp32')
                report['coverage']=coverage(report['samples']);write_json(partial,report)
            final=report['coverage']
            if (final['paths'],final['null_scored'],final['null_targets'],final['completed_cases'])!=(1536,426,5024,12):
                raise ValueError('Incomplete final denominator or null-score coverage')
            verify_inputs(hashes);stop_check()
            report.update(complete=True,seconds=time.monotonic()-started,input_pre_post_sha_equal=True)
            write_json(partial,report)
            if output.exists():raise FileExistsError(output)
            partial.rename(output)
        finally:shared.AUDIT=previous
    return dict(output=output.relative_to(ROOT).as_posix(),coverage=report['coverage'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('audit/rhyme17/word_rank/allword_native_rescore.json'))
    parser.add_argument('--panel',type=Path,default=Path('audit/rhyme17/simple12_panel.json'))
    parser.add_argument('--models',type=Path,default=Path('audit/rhyme17/diagnostic_models.json'))
    parser.add_argument('--output',type=Path,default=Path('audit/rhyme17/word_rank/context_null_rescore.json'))
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    args=parser.parse_args()
    print(run(args.source,args.panel,args.models,args.output,args.device))

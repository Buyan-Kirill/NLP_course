"""Pinned native poetry system + common free baseline on visible inputs only.

No method/model selection or sealed-file opening. The selected system is an
inference algorithm, not a claim about free LM ability or semantic correctness.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tokenizers import Tokenizer
from poetrygpt.data import read_json,sha256
from poetrygpt.generate import generate
from poetrygpt.model import GPT,ModelConfig
from poetrygpt.runtime import load_checkpoint,run_lock,write_json,seed_everything
from poetrygpt.serialization import special_ids
from scripts import prepare_rhyme17_runtime_pool as pool_api,rerank_rhyme17_context as context_core
from scripts.rhyme17_native_joint import search
from scripts.rhyme_endings16 import VOWELS

SOURCES=tuple(sorted(set(context_core.SOURCES)|{
    'scripts/generate_rhyme17_system.py','scripts/prepare_rhyme17_runtime_pool.py',
    'scripts/rhyme17_native_joint.py'}))


def local(path):
    path=pool_api.local(path)
    if any('sealed' in part.casefold() or part.casefold()=='final16' for part in path.relative_to(ROOT).parts):
        raise ValueError('Use a separately opened visible copy, never original sealed/final16 paths')
    return path


def recipe(selector):
    if selector not in ('native','context_null'):raise ValueError('Explicit native/context_null selector required')
    return dict(method='native_joint',selector=selector,include_free_baseline=True,
        seed=7,body_limit=24,temperature=.7,top_k=40,top_p=.9,
        width=8,branch=8,max_prefix_tokens=16,min_prefix_words=2,
        syllable_tolerance=2,top_paths=128,max_pool_words=32,
        native_gap=.5,null_weight=1/3)


def line_projection(raw,stop_reason):
    """First nonempty physical line as exact source slice, never repaired."""
    for match in re.finditer(r'[^\r\n]*(?:\r\n|\r|\n|$)',raw):
        text=match.group()
        if text.rstrip('\r\n').strip():
            separator=text.endswith(('\r','\n'))
            complete=separator or stop_reason=='stop_token'
            return dict(text=text,source_span=[match.start(),match.end()],complete=complete,
                tail=raw[match.end():],status='complete' if complete else 'unterminated_fragment')
    return dict(text=None,source_span=None,complete=False,tail=None,status='empty')


def free_baseline(model,case,token,*,precision):
    special=special_ids(token)
    stops=[special['<|eos|>'],special['<|eot|>']]
    forbidden=[special[k] for k in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>')]
    effective=7+int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16)
    result=dict(status='not_attempted',effective_seed=effective,raw_text=None,generated_ids=None,
        generated_tokens=0,stop_reason=None,stop_token=None,projection=None,error=None)
    try:
        seed_everything(effective)
        ids,reason=generate(model,case['context_ids'],max_new_tokens=24,temperature=.7,top_k=40,top_p=.9,
            stop_ids=stops,forbidden_ids=forbidden,precision=precision)
        result.update(generated_ids=ids,generated_tokens=len(ids),stop_reason=reason)
        if (not isinstance(ids,list) or not ids or len(ids)>24
                or any(type(t) is not int or not 0<=t<token.get_vocab_size() for t in ids)
                or any(t in forbidden for t in ids) or any(t in stops for t in ids[:-1])
                or reason not in ('stop_token','max_new_tokens')
                or (reason=='stop_token')!=(ids[-1] in stops)
                or (reason=='max_new_tokens' and len(ids)!=24)):
            raise ValueError('Invalid free sampler IDs/stop/budget')
        raw=token.decode(ids,skip_special_tokens=True)
        result.update(status='generated' if raw.strip() else 'empty_output',raw_text=raw,
            stop_token=token.decode([ids[-1]],skip_special_tokens=False) if ids[-1] in stops else None,
            projection=line_projection(raw,reason))
    except Exception as error:
        result.update(status='generation_error',error=f'{type(error).__name__}: {error}')
    return result


def pool_words(case,pool):
    if not isinstance(pool,list) or len(pool)>32:raise ValueError('At most32 pool entries required')
    first,_=pool_api.first_endword(case['first_line'])
    words=[];seen=set()
    for entry in pool:
        word=entry['word'];fold=word.casefold().replace('ё','е')
        if (not re.fullmatch('[А-Яа-яЁё]+',word) or fold in seen or fold==pool_api.fold(first or '')
                or entry['phonetic']['exact_all'] is not True):
            raise ValueError('Distinct whole noncompound exact-rhyme pool words required')
        words.append(word);seen.add(fold)
    return words


def run_case(model,case,pool,token,selector,device,precision):
    """Pure case inference: only5 visible fields and candidate word/phonetic."""
    visible=pool_api.project_cases(dict(cases=[case]),token)[0]
    recipe(selector)
    words=pool_words(visible,pool)
    result=dict(case=visible,method='native_joint',selector=selector,pool_words=words,status='not_attempted',
        free=None,selected=None,search=None,error=None)
    if len(visible['context_ids'])+24>model.config.context_length:
        result['status']='not_attempted_context_budget'
        return result
    result['free']=free_baseline(model,visible,token,precision=precision)
    if result['free']['status']=='generation_error':
        result['status']='free_baseline_error'
        return result
    if not words:
        result['status']='no_pool_candidates'
        return result
    begin=time.monotonic()
    try:
        syllables=sum(c.lower() in VOWELS for c in visible['first_line'])
        built=search(model,visible['context_ids'],words,token,device=device,precision=precision,
            forbidden_ids=list(special_ids(token).values()),syllable_range=[max(1,syllables-2),syllables+2],
            width=8,branch=8,max_prefix_tokens=16,body_limit=24,min_prefix_words=2)
        paths=built.pop('candidates')
        digest=hashlib.sha256(json.dumps(paths,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        retained=sorted(paths,key=lambda r:(r['mean_logp'],r['logp_sum'],r['text']),reverse=True)[:128]
        result['search']=dict(built,completed_candidates=len(paths),all_paths_sha256=digest,
            retained_candidates=len(retained),omitted_candidates=max(0,len(paths)-128),
            null_scored_paths=0,null_target_positions=0,seconds=0,
            compute_caveat='scored_body_positions excludes prefix-forward and padding cost; seconds measures elapsed inference')
        if not retained:
            result['status']='no_complete_candidates'
        else:
            adapted=dict(case_id=visible['id'],first_line=visible['first_line'],candidates=retained,
                native_no_repeats_selected=built['selected'])
            prepared=context_core.prepare_case(adapted,visible,token)
            if selector=='context_null':
                prepared=context_core.score_case(model,prepared,device=device,precision=precision)
                result['search']['null_scored_paths']=sum(r['null_scored'] for r in prepared['candidates'])
                result['search']['null_target_positions']=sum(r['null_targets'] for r in prepared['candidates'])
                chosen=prepared['selected']
            else:chosen=deepcopy(built['selected'])
            result['search']['candidates']=prepared['candidates']
            result['selected']=chosen
            result['status']='selected' if chosen is not None else 'no_selector_candidates'
        result['search']['seconds']=time.monotonic()-begin
    except Exception as error:
        result.update(status='search_error',error=f'{type(error).__name__}: {error}')
    return result


def load_contract(config_path,input_path,pool_path,pool_sha256):
    """Hash-bound CPU preflight. Authorizations assert opening; never open seal."""
    frozen={}
    def checked_read(path):
        path=local(path);before=sha256(path);value=read_json(path)
        if sha256(path)!=before:raise ValueError('Changed input during read')
        frozen[path]=before
        return value
    config=checked_read(config_path)
    if config.get('schema')!='rhyme17-generation-system-1':raise ValueError('Pinned runtime config required')
    if config['recipe']!=recipe(config['recipe']['selector']):raise ValueError('Changed/unpinned recipe')
    for key in ('checkpoint','tokenizer','vocabulary'):
        reference=config[key]
        if set(reference)!={'path','sha256'}:raise ValueError('Explicit path/fullSHA reference required')
        path=local(reference['path'])
        if not pool_api.full_sha(reference['sha256']) or sha256(path)!=reference['sha256']:raise ValueError('Pinned '+key+' SHA mismatch')
        frozen[path]=reference['sha256']
    if set(config['code_sha256'])!=set(SOURCES):raise ValueError('Complete runtime code identity required')
    for name,digest in config['code_sha256'].items():
        path=local(name)
        if sha256(path)!=digest:raise ValueError('Pinned runtime code changed')
        frozen[path]=digest
    if not pool_api.full_sha(pool_sha256) or sha256(local(pool_path))!=pool_sha256:
        raise ValueError('Explicit expected pool SHA mismatch')
    pool=checked_read(pool_path)
    if frozen[local(pool_path)]!=pool_sha256:raise ValueError('Pool changed during read')
    if (pool.get('schema')!='rhyme17-runtime-pool-1' or pool.get('complete') is not True
            or pool.get('gold_ending_or_reference_used') is not False or pool['policy']['max_choices']!=32):
        raise ValueError('Completed non-oracle runtime phonetic pool required')
    if (any(pool['policy'].get(k) is not True for k in ('nonhyphen_only','exact_all_required','yo_e_dedup'))
            or type(pool['test_used']) is not bool or pool['split'] not in ('open_dev','inference','test_opened')
            or pool['test_used']!=(pool['split']=='test_opened')):
        raise ValueError('Pool policy or split/test_used mismatch')
    if (pool['tokenizer_sha256']!=config['tokenizer']['sha256'] or pool['vocabulary_sha256']!=config['vocabulary']['sha256']):
        raise ValueError('Pool/config tokenizer or vocabulary mismatch')
    # Authorize metadata BEFORE reading any test-opened visible case contents.
    input_path=local(input_path)
    input_sha=sha256(input_path)
    if input_sha!=pool['input_sha256']:raise ValueError('Pool/input SHA mismatch')
    opening=pool['opening_authorization']
    if pool['split']=='test_opened':
        if not isinstance(opening,dict):raise ValueError('Opened-test authorization missing')
        auth=checked_read(opening['path'])
        if (frozen[local(opening['path'])]!=opening['sha256']
                or auth.get('runtime_config_sha256')!=frozen[local(config_path)]):
            raise ValueError('Opened-test authorization not bound to selected runtime config')
    elif opening is not None:raise ValueError('Opening authorization cannot relabel open input')
    payload=checked_read(input_path)
    if (payload.get('schema')!='rhyme17-visible-runtime-input-1'
            or frozen[input_path]!=input_sha or payload['split']!=pool['split'] or payload['test_used'] is not pool['test_used']
            or payload['tokenizer_sha256']!=config['tokenizer']['sha256']):
        raise ValueError('Visible input identity/split drift')
    verified=pool_api.opening_metadata(payload,input_sha,opening['path'] if opening else None)
    if opening:
        for key in ('path','sha256','original_seal_sha256','selection_sha256','runtime_config_sha256','frozen_before_heldout'):
            if verified[key]!=opening[key]:raise ValueError('Opening metadata differs from actual authorization')
    token=Tokenizer.from_file(str(local(config['tokenizer']['path'])))
    cases=pool_api.project_cases(payload,token)
    pool_cases=pool_api.project_cases(dict(cases=pool['results']),token)
    if cases!=pool_cases:raise ValueError('Exact visible context/order differs from pool')
    allowed_status={'ok','no_candidates','unsupported_first_endword','unknown_first_stress','phonetic_error'}
    for row in pool['results']:
        if (row['case_id']!=row['id'] or row['status'] not in allowed_status or len(row['candidates'])>32
                or (row['status']!='ok' and row['candidates'])):raise ValueError('Invalid case pool status/coverage')
        pool_words(row,row['candidates'])
    producer_sources=pool['sources_sha256']
    required_producer=set(pool_api.SOURCES)|{'artifacts/evaluators/rpst/version.json'}
    if not required_producer.issubset(producer_sources):raise ValueError('Producer builder/probe/version identity missing')
    for name,digest in producer_sources.items():
        path=local(name)
        if not pool_api.full_sha(digest):raise ValueError('Invalid producer source SHA')
        # Pool transport binds all producer evidence. Only code/version metadata
        # is independently reverified here; RPST weights/raw corpus are not used.
        if path.suffix!='.py' and name not in required_producer:continue
        if sha256(path)!=digest:raise ValueError('Pool origin changed')
        if path in frozen and frozen[path]!=digest:raise ValueError('Conflicting input identities')
        frozen[path]=digest
    verify(frozen)
    return config,payload,pool,cases,token,frozen


def verify(frozen):
    for path,digest in frozen.items():
        if sha256(path)!=digest:raise ValueError('Runtime input changed: '+str(path))


def coverage(samples):
    return dict(cases=len(samples),selected=sum(r.get('status')=='selected' for r in samples),
        statuses={s:sum(r.get('status')==s for r in samples) for s in sorted({r.get('status') for r in samples})},
        free_complete_lines=sum(bool(r.get('free') and r['free'].get('projection') and r['free']['projection']['complete']) for r in samples))


def select_cases(cases,entries,case_ids,split):
    if case_ids is None:return cases,entries
    if split=='test_opened':raise ValueError('Opened-test runtime always processes all cases; subset forbidden')
    if not 1<=len(case_ids)<=12 or len(set(case_ids))!=len(case_ids):raise ValueError('Explicit unique subset of1..12 case IDs required')
    by_id={case['id']:(case,entry) for case,entry in zip(cases,entries)}
    if any(cid not in by_id for cid in case_ids):raise ValueError('Unknown subset case ID')
    selected=[by_id[cid] for cid in case_ids]
    return [r[0] for r in selected],[r[1] for r in selected]


def run(config_path,input_path,pool_path,output,device_name,*,pool_sha256,case_ids=None):
    config_path,input_path,pool_path,output=map(local,(config_path,input_path,pool_path,output))
    partial=output.with_suffix('.incomplete.partial.json')
    with run_lock(ROOT/'runs/pretrain_series01','series'):
        if output.exists() or partial.exists():raise FileExistsError('Preserve existing output/partial')
        def stop_check():
            if (ROOT/'runs/pretrain_series01/STOP').exists() or (output.parent/'STOP').exists():raise RuntimeError('STOP requested')
        stop_check()
        cfg,payload,pool,cases,token,frozen=load_contract(config_path,input_path,pool_path,pool_sha256)
        full_source_case_count=len(cases)
        cases,entries=select_cases(cases,pool['results'],case_ids,payload['split'])
        report=dict(schema='rhyme17-visible-system-generation-1',complete=False,generation=True,training=False,oracle=False,
            device=device_name,precision='bf16' if device_name=='cuda' else 'fp32',
            split=payload['split'],test_used=payload['test_used'],opening_authorization=pool['opening_authorization'],
            config_sha256=frozen[config_path],recipe=cfg['recipe'],checkpoint=cfg['checkpoint'],
            tokenizer=cfg['tokenizer'],vocabulary=cfg['vocabulary'],
            runtime_verified_inputs_sha256={p.relative_to(ROOT).as_posix():h for p,h in frozen.items()},
            producer_verified_sources_sha256=pool['sources_sha256'],
            producer_sources_not_reverified_on_this_device={n:h for n,h in pool['sources_sha256'].items() if local(n) not in frozen},
            provenance_boundary='Producer verified origin hashes retained through exact expected pool SHA; runtime independently verifies actual inputs, all executed source code, producer code and RPST version metadata, not RPST weights or raw training corpus.',
            full_source_case_count=full_source_case_count,selected_case_ids=[c['id'] for c in cases],
            limitation='Selected native inference system, not free LM quality. EOS-null contrast is not true PMI or a semantic judge. Test opening is a hash-bound audit assertion, not cryptographic access control.',
            samples=[dict(case=c,status='not_attempted',free=None,selected=None) for c in cases])
        report['coverage']=coverage(report['samples'])
        output.parent.mkdir(parents=True,exist_ok=True);write_json(partial,report)
        from scripts import shared_gpu11 as shared
        previous=shared.AUDIT;begin=time.monotonic()
        try:
            shared.AUDIT=output.with_suffix('.gpu_admission.jsonl')
            if device_name=='cuda':shared.admit(torch,3072,shared.Audit(phase='rhyme17_visible_system',config_sha256=frozen[config_path]))
            verify(frozen);stop_check()
            state=load_checkpoint(local(cfg['checkpoint']['path']))
            if state['tokenizer_sha256']!=cfg['tokenizer']['sha256']:raise ValueError('Native checkpoint tokenizer mismatch')
            report['native_checkpoint_metadata']=dict(format=state['format'],lineage=state['lineage'],
                model_config=state['model_config'],tokens_seen=state.get('tokens_seen'),step=state.get('step'))
            model=GPT(ModelConfig(**state['model_config']));model.load_state_dict(state['model'],strict=True)
            device=torch.device(device_name);model.to(device).eval();del state;torch.set_num_threads(4)
            for index,(case,entry) in enumerate(zip(cases,entries)):
                stop_check()
                row=run_case(model,case,entry['candidates'],token,cfg['recipe']['selector'],device,'bf16' if device_name=='cuda' else 'fp32')
                row['pool_status']=entry['status'];row['pool_failure_reason']=entry['failure_reason']
                report['samples'][index]=row;report['coverage']=coverage(report['samples']);write_json(partial,report)
                if row['status']=='search_error' or (row.get('free') and row['free']['status']=='generation_error'):
                    raise RuntimeError('Inference error retained in partial; no silent retry')
            verify(frozen);stop_check()
            if len(report['samples'])!=len(cases) or any(r['status']=='not_attempted' for r in report['samples']):raise ValueError('Incomplete case coverage')
            report.update(complete=True,seconds=time.monotonic()-begin,input_pre_post_sha_equal=True)
            report['coverage']=coverage(report['samples']);write_json(partial,report)
            os.link(partial,output)  # Atomic creation, refuses an existing destination.
            partial.unlink()
        finally:shared.AUDIT=previous
    return dict(output=output.relative_to(ROOT).as_posix(),coverage=report['coverage'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','input','pool','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--pool-sha256',required=True)
    parser.add_argument('--case-ids',nargs='+',help='Explicit open_dev/inference subset only; opened tests always all cases')
    args=parser.parse_args()
    print(run(args.config,args.input,args.pool,args.output,args.device,pool_sha256=args.pool_sha256,case_ids=args.case_ids))

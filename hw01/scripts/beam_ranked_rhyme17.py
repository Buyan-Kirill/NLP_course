"""Dictionary-only contextual whole-word ranking before frozen form-beam-v2.

Reads a visible case projection only; no supplied/oracle/reference-ending mode.
Extra pool-ranking forwards are explicit additional inference compute.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path,PureWindowsPath
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tokenizers import Tokenizer
from poetrygpt.data import read_json,sha256
from poetrygpt.generate import generate
from poetrygpt.model import GPT,ModelConfig
from poetrygpt.runtime import load_checkpoint,run_lock,seed_everything,write_json
from poetrygpt.serialization import special_ids
from scripts.evaluate_ending_plan16 import parse_output,token_budget
from scripts.prepare_ending_plan16 import PLAN_MARKER
from scripts.rhyme17_beam_form import search
from scripts.rhyme17_search import score_candidates,select_candidate,WORD
from scripts.rhyme_endings16 import VOWELS,last_word

CASE_FIELDS=('id','group_id','context_ids','context_text','first_line')
SOURCES=('scripts/beam_ranked_rhyme17.py','scripts/rhyme17_beam_form.py','scripts/rhyme17_search.py',
    'scripts/rhyme_endings16.py','scripts/evaluate_ending_plan16.py','scripts/prepare_ending_plan16.py',
    'poetrygpt/generate.py','poetrygpt/model.py','poetrygpt/runtime.py','poetrygpt/serialization.py',
    'poetrygpt/data.py','scripts/shared_gpu11.py')


def visible_cases(panel,tokenizer):
    # Do not iterate/access optional case values such as ending/reference/options.
    cases=[{field:deepcopy(row[field]) for field in CASE_FIELDS} for row in panel['cases']]
    if not cases or len(cases)>64 or len({c['id'] for c in cases})!=len(cases):
        raise ValueError('Empty/duplicate/unbounded visible case IDs')
    for case in cases:
        ids=case['context_ids'];text=case['context_text']
        if (not isinstance(case['id'],str) or not case['id'] or not isinstance(case['group_id'],str)
                or not case['group_id'] or not ids
                or any(type(i) is not int or not 0<=i<tokenizer.get_vocab_size() for i in ids)):
            raise ValueError('Invalid visible case identity/prefix')
        visible=tokenizer.decode(ids,skip_special_tokens=True)
        lines=[line for line in visible.splitlines() if line.strip()]
        if (tokenizer.decode(ids,skip_special_tokens=False)!=text or not text.endswith('\n')
                or tokenizer.encode(text,add_special_tokens=False).ids!=ids
                or not lines or lines[-1]!=case['first_line']):
            raise ValueError('Visible C/first_line exact roundtrip mismatch')
    return cases


def bind_pools(cases,dictionary):
    if (dictionary['gold_ending_or_reference_used'] is not False
            or dictionary['closed_reserve_used'] is not False
            or dictionary['training_test_data_used'] is not False
            or dictionary['case_selection_fields']!=['id','first_line']):
        raise ValueError('Require train-only non-oracle pool')
    rows=dictionary['results'];index={r['case_id']:r for r in rows}
    if len(index)!=len(rows) or set(index)!={c['id'] for c in cases}:
        raise ValueError('Duplicate/missing/extra dictionary case IDs')
    result={}
    for case in cases:
        row=index[case['id']]
        if row['first_line']!=case['first_line']:raise ValueError('Stale dictionary first_line binding')
        entries=row['candidates'];seen=set();pool=[]
        if not 1<=len(entries)<=32:raise ValueError('Expected nonempty pool of at most32')
        for rank,entry in enumerate(entries,1):
            word=entry['word'];key=word.casefold().replace('ё','е')
            if (WORD.fullmatch(word) is None or key in seen
                    or key==(last_word(case['first_line']) or '').replace('ё','е')
                    or entry['phonetic']['exact_all'] is not True):
                raise ValueError('Duplicate/non-whole/non-exact/identical ending in pool')
            seen.add(key)
            pool.append(dict(word=word,source_pool_rank=rank,
                freq=entry['freq'],workcount=entry['workcount']))
        result[case['id']]=pool
    return result


def ranking_prefix(case,tokenizer):
    text=case['context_text']+PLAN_MARKER
    ids=tokenizer.encode(text,add_special_tokens=False).ids
    if ids[:len(case['context_ids'])]!=case['context_ids']:
        raise ValueError('Ranking marker retokenized C')
    if tokenizer.decode(ids,skip_special_tokens=False)!=text:raise ValueError('Ranking prefix roundtrip')
    return ids,text


def full_word_candidates(prefix,prefix_text,pool,tokenizer):
    rows=[]
    for entry in pool:
        text=entry['word']+'\n'
        ids=tokenizer.encode(prefix_text+text,add_special_tokens=False).ids
        if ids[:len(prefix)]!=prefix:raise ValueError('Word boundary retokenized ranking prefix')
        body=ids[len(prefix):]
        if not body or tokenizer.decode(ids,skip_special_tokens=False)!=prefix_text+text:
            raise ValueError('Whole W+newline roundtrip failure')
        rows.append(dict(entry,text=text,body_ids=body,ending=entry['word']))
    return rows


def rank_words(model,case,pool,tokenizer,*,device,precision):
    if model.training:raise ValueError('Word ranker requires eval mode')
    prefix,text=ranking_prefix(case,tokenizer)
    candidates=full_word_candidates(prefix,text,pool,tokenizer)
    scored=score_candidates(model,prefix,candidates,device=device,precision=precision)
    # Same untempered mean LP as required; explicit deterministic tie-break,
    # independent of input frequency order and all reference fields.
    scored.sort(key=lambda r:(r['mean_logp'],r['logp_sum'],r['word']),reverse=True)
    for rank,row in enumerate(scored,1):row.update(context_rank=rank,selected_top4=rank<=4)
    return dict(prefix_ids=prefix,prefix_text=text,ranked=scored,
        selected_words=[r['word'] for r in scored[:4]],
        scored_W_positions=sum(r['targets'] for r in scored),candidates_scored=len(scored),
        ranking_forward_batches=(len(candidates)+15)//16,
        ranking_padded_input_positions=sum(len(chunk)*max(len(prefix)+len(r['body_ids']) for r in chunk)
            for chunk in (candidates[i:i+16] for i in range(0,len(candidates),16))))


def check_beam_roundtrip(beam,prompt,prompt_text,tokenizer):
    for row in beam['candidates']:
        full=prompt+row['body_ids']
        if (tokenizer.decode(full,skip_special_tokens=False)!=prompt_text+row['text']
                or tokenizer.encode(prompt_text+row['text'],add_special_tokens=False).ids!=full):
            raise ValueError('Beam full body roundtrip failed')


def decode_case(model,case,pool,tokenizer,*,device,precision):
    ranking=rank_words(model,case,pool,tokenizer,device=device,precision=precision)
    special=special_ids(tokenizer)
    stops=[special[s] for s in ('<|eos|>','<|eot|>')]
    forbidden=[special[s] for s in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>')]
    syllables=sum(c.lower() in VOWELS for c in case['first_line'])
    band=[max(1,syllables-2),syllables+2];alternatives=[]
    effective_seed=7+int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16)
    for word in ranking['selected_words']:
        prompt_text=case['context_text']+PLAN_MARKER+word+'\n'
        prompt=tokenizer.encode(prompt_text,add_special_tokens=False).ids
        if (prompt[:len(case['context_ids'])]!=case['context_ids']
                or tokenizer.decode(prompt,skip_special_tokens=False)!=prompt_text
                or len(prompt)+24>model.config.context_length):
            raise ValueError('Changed/oversized conditioned prompt')
        seed_everything(effective_seed)
        ids,reason=generate(model,prompt,24,.7,40,stops,forbidden,precision,top_p=.9)
        raw=tokenizer.decode(ids,skip_special_tokens=True)
        parsed=parse_output(raw,'normal',reason)
        budget=token_budget(ids,raw,'normal',reason,tokenizer,24,0)
        baseline=[]
        if parsed['complete'] and budget['eligible']:
            body=ids[:budget['body_tokens']]
            scored_text=tokenizer.decode(body,skip_special_tokens=False)
            if tokenizer.decode(prompt+body,skip_special_tokens=False)!=prompt_text+scored_text:
                raise ValueError('Sampled body decode roundtrip failed')
            baseline=score_candidates(model,prompt,[dict(text=parsed['line'],body_ids=body,ending=word,
                requested_ending=word,actual_ending=last_word(parsed['line']),
                ending_compliance=last_word(parsed['line'])==word.lower(),
                scored_body_text=scored_text,
                sampled_ids_canonical_reencoding=(tokenizer.encode(prompt_text+scored_text,add_special_tokens=False).ids==prompt+body))],
                device=device,precision=precision)
        beam=search(model,prompt,word,tokenizer,device=device,precision=precision,
            forbidden_ids=forbidden+stops,width=8,branch=8,max_prefix_tokens=16,
            body_limit=24,min_prefix_words=2,word_ngram=2,syllable_range=band)
        check_beam_roundtrip(beam,prompt,prompt_text,tokenizer)
        alternatives.append(dict(ending=word,prompt_ids=prompt,baseline_raw=raw,baseline_parsed=parsed,
            baseline_generated_ids=ids,baseline_generated_tokens=len(ids),baseline_stop_reason=reason,
            baseline_token_budget=budget,baseline=baseline[0] if baseline else None,beam=beam,syllable_band=band))
    samples=[]
    for kind in ('baseline','beam'):
        options=[a[kind] if kind=='baseline' else a[kind]['selected'] for a in alternatives]
        selected=select_candidate([row for row in options if row is not None])
        samples.append(dict(case_id=case['id'],group_id=case['group_id'],first_line=case['first_line'],
            mode='dictionary',kind=kind,oracle=False,selected=selected,attempted_endings=ranking['selected_words']))
    return dict(case=case,ranking=ranking,effective_seed=effective_seed,alternatives=alternatives,samples=samples,
        extra_ranking_compute=dict(pool_candidates=ranking['candidates_scored'],W_newline_target_positions=ranking['scored_W_positions'],
            forward_batches=ranking['ranking_forward_batches'],padded_input_positions=ranking['ranking_padded_input_positions'],
            note='Additional full-prompt forwards for pool ranking; not included in frozen beam scored_body_positions'))


def local(path):
    path=(ROOT/path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Require project-local path')
    return path


def dictionary_source_path(recorded):
    """Relocate historical producer-absolute provenance within this project.

    Keep the original source key in the report; the SHA must still match the
    actual project file. Never require the producer's Windows home on a worker.
    """
    parts=PureWindowsPath(recorded).parts
    if not PureWindowsPath(recorded).is_absolute():return local(recorded)
    if parts.count(ROOT.name)!=1:raise ValueError('Dictionary source is not a project provenance path')
    suffix=parts[parts.index(ROOT.name)+1:]
    if not suffix or suffix[0] not in ('data','scripts','audit','artifacts'):
        raise ValueError('Unsupported dictionary source location')
    return local(Path(*suffix))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','models','dictionary','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--label',required=True)
    args=p.parse_args()
    panel_path,models_path,pool_path,output=map(local,(args.panel,args.models,args.dictionary,args.output))
    partial=output.with_suffix('.incomplete.partial.json')
    # No reserve paths, no automatic opening, no supplied/oracle CLI switch.
    if any(panel_path.is_relative_to(local(name)) for name in ('audit/rhyme17/chat_reserve17',
            'audit/rhyme17/chat_reserve17_final','audit/rhyme17/reserve','audit/sft16/reserve')):
        raise ValueError('Open diagnostic panels only')
    with run_lock(ROOT/'runs/pretrain_series01','series'):
        if output.exists() or partial.exists():raise FileExistsError('Preserve existing output/partial')
        if any(path.exists() for path in (ROOT/'runs/pretrain_series01/STOP',output.parent/'STOP')):
            raise RuntimeError('STOP requested')
        specs=read_json(models_path);matches=[s for s in specs['models'] if s['label']==args.label]
        if len(matches)!=1:raise ValueError('Choose exactly one explicit model')
        spec=matches[0];checkpoint=local(spec['checkpoint'])
        token_path=ROOT/'artifacts/tokenizers_v3/selected/tokenizer.json'
        frozen={path:sha256(path) for path in (panel_path,models_path,pool_path,token_path,*[ROOT/n for n in SOURCES])}
        frozen[checkpoint]=spec['sha256']
        def verify():
            for path,digest in frozen.items():
                if sha256(path)!=digest:raise ValueError('Changed frozen input: '+str(path))
        verify();tokenizer=Tokenizer.from_file(str(token_path))
        # Only the five visible fields are projected from the panel cases.
        panel=read_json(panel_path)
        if panel.get('test_used') is not False:raise ValueError('Explicitly open diagnostic panel required')
        cases=visible_cases(panel,tokenizer)
        dictionary=read_json(pool_path);pools=bind_pools(cases,dictionary)
        source_locations={}
        for name,digest in dictionary['sources_sha256'].items():
            path=dictionary_source_path(name)
            if sha256(path)!=digest:raise ValueError('Dictionary source SHA changed')
            frozen[path]=digest
            source_locations[name]=str(path.relative_to(ROOT))
        verify()
        report=dict(schema='rhyme17-context-ranked-dictionary-form-beam-1',complete=False,training=False,test_used=False,
            label=args.label,oracle=False,case_projection_fields=list(CASE_FIELDS),
            checkpoint_sha256=spec['sha256'],panel_sha256=frozen[panel_path],dictionary_sha256=frozen[pool_path],
            tokenizer_sha256=frozen[token_path],inputs_sha256={str(path.relative_to(ROOT)):digest for path,digest in frozen.items()},
            dictionary_provenance_sources_sha256=dictionary['sources_sha256'],dictionary_source_locations=source_locations,
            ranking=dict(objective='mean untempered LP of whole W+newline after exact C+PLAN_MARKER',
                tie_break='descending mean_logp, logp_sum, word; no frequency or oracle tie-break',pool_cap=32,top_k_words=4),
            decoding=dict(width=8,branch=8,max_prefix_tokens=16,body_limit=24,syllable_slack=2,
                word_ngram=2,temperature=.7,top_k=40,top_p=.9,seed=7,
                final_selection='unchanged mean body LP through select_candidate across up to4 endings'),
            samples=[],searches=[])
        from scripts import shared_gpu11 as shared
        old_audit=shared.AUDIT;started=time.monotonic()
        try:
            shared.AUDIT=output.with_suffix('.gpu_admission.jsonl')
            shared.admit(torch,3072,shared.Audit(phase='rhyme17_context_word_rank',evaluator_sha256=sha256(__file__)))
            state=load_checkpoint(checkpoint)
            if state['tokenizer_sha256']!=frozen[token_path]:raise ValueError('Tokenizer mismatch')
            report['model']=state['model_config']
            report['checkpoint_identity']={key:state[key] for key in ('step','tokens_seen','fingerprint') if key in state}
            model=GPT(ModelConfig(**state['model_config']));model.load_state_dict(state['model'])
            model.to('cuda').eval();del state;torch.set_num_threads(4)
            output.parent.mkdir(parents=True,exist_ok=True);write_json(partial,report)
            for case in cases:
                result=decode_case(model,case,pools[case['id']],tokenizer,device=torch.device('cuda'),precision='bf16')
                report['searches'].append(result);report['samples'].extend(result['samples']);write_json(partial,report)
                print(json.dumps(dict(case_id=case['id'],seconds=time.monotonic()-started)),flush=True)
            expected={(c['id'],kind) for c in cases for kind in ('baseline','beam')}
            actual=[(s['case_id'],s['kind']) for s in report['samples']]
            if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Coverage mismatch')
            verify();report.update(complete=True,seconds=time.monotonic()-started)
            write_json(partial,report)
            if output.exists():raise FileExistsError(output)
            partial.rename(output)
        finally:shared.AUDIT=old_audit


if __name__=='__main__':main()

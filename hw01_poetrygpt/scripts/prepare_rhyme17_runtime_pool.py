"""Reproducible CPU phonetic pools from visible inputs and frozen train words.

No references/desired endings are accessed. Opening authorization is an audit
assertion bound to hashes, not cryptographic access control or proof of abstention.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tokenizers import Tokenizer
from poetrygpt.data import read_json,sha256
from poetrygpt.runtime import run_lock,write_json

TOKENIZER=ROOT/'artifacts/tokenizers_v3/selected/tokenizer.json'
TOKENIZER_SHA='4cd18c31a8a5879ee3eacaf0936c191de011a71aed7fd13812f1d7b5d7e9d1fe'
VOCAB=ROOT/'data/rhyme17_candidate/vocabulary_all_poetry_wordforms.json'
VOCAB_SHA='76fecc9e9d94b65ec9f8f87c07b8f0870f7c810b621c6bc14d600896240cad85'
ARTIFACTS=ROOT/'artifacts/evaluators/rpst'
RPST_REVISION='9e577f036d970ba648f19d9d0c286547d9327902'
FIELDS=('id','group_id','first_line','context_text','context_ids')
WORD=re.compile(r'(?<![А-Яа-яЁёA-Za-z0-9_\-‐‑])[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*(?![А-Яа-яЁёA-Za-z0-9_\-‐‑])')
SOURCES=('scripts/prepare_rhyme17_runtime_pool.py','scripts/rhyme_endings16.py',
    'scripts/probe_rpst08.py','poetrygpt/data.py','poetrygpt/runtime.py')


def local(path):
    path=(ROOT/path).resolve()
    if path==ROOT or not path.is_relative_to(ROOT):raise ValueError('Project-local path required')
    if any('reserve' in p.casefold() or 'closed' in p.casefold() for p in path.relative_to(ROOT).parts):
        raise ValueError('Sealed/reserve paths are forbidden; use an explicitly opened copy elsewhere')
    return path


def full_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def fold(word):return word.casefold().replace('ё','е')
def relative(path):return path.relative_to(ROOT).as_posix()


def project_cases(payload,tokenizer):
    # Whitelist lookups, never optional values such as W, ending or reference.
    projected=[{k:deepcopy(case[k]) for k in FIELDS} for case in payload['cases']]
    if not projected:raise ValueError('Nonempty case list required')
    for case in projected:
        if any(not isinstance(case[k],str) or not case[k].strip() for k in ('id','group_id','first_line','context_text')):
            raise ValueError('Nonempty visible strings required')
        ids=case['context_ids'];text=case['context_text'];first=case['first_line']
        if (not isinstance(ids,list) or not ids or any(type(t)is not int or not 0<=t<tokenizer.get_vocab_size() for t in ids)
                or '\n' in first or '\r' in first or not text.endswith('\n')
                or tokenizer.decode(ids,skip_special_tokens=False)!=text
                or tokenizer.encode(text,add_special_tokens=False).ids!=ids):
            raise ValueError('Malformed/canonical C token IDs or first_line')
        lines=[line for line in tokenizer.decode(ids,skip_special_tokens=True).splitlines() if line.strip()]
        if not lines or lines[-1]!=first:raise ValueError('C last visible line must exactly match first_line')
    if len({r['id'] for r in projected})!=len(projected):raise ValueError('Unique case IDs required')
    return projected


def opening_metadata(payload,input_sha,authorization_path):
    split=payload['split'];used=payload['test_used']
    if type(used)is not bool or split not in ('open_dev','inference','test_opened') or used!=(split=='test_opened'):
        raise ValueError('Explicit consistent split/test_used required')
    if split!='test_opened':
        if authorization_path is not None:raise ValueError('Opening authorization only for test_opened')
        return None
    if authorization_path is None:raise ValueError('Opened test input requires explicit authorization')
    authorization_path=local(authorization_path)
    auth=read_json(authorization_path)
    if (auth.get('schema')!='rhyme17-opened-inference-authorization-1'
            or auth.get('authorized') is not True or auth.get('frozen_before_heldout') is not True
            or auth.get('opened_input_sha256')!=input_sha
            or auth.get('tokenizer_sha256')!=TOKENIZER_SHA
            or not full_sha(auth.get('original_seal_sha256')) or not full_sha(auth.get('selection_sha256'))
            or not full_sha(auth.get('runtime_config_sha256'))
            or payload.get('original_seal_sha256')!=auth['original_seal_sha256']):
        raise ValueError('Opening authorization/input/tokenizer/seal/selection mismatch')
    return dict(path=relative(authorization_path),sha256=sha256(authorization_path),
        original_seal_sha256=auth['original_seal_sha256'],selection_sha256=auth['selection_sha256'],
        runtime_config_sha256=auth['runtime_config_sha256'],
        frozen_before_heldout=True,meaning='Explicit audit assertion, not cryptographic access control')


def first_endword(first):
    clean=first.casefold().replace('\u0301','').replace('\u0341','')
    matches=list(WORD.finditer(clean))
    if not matches:return None,'missing_whole_cyrillic_ending'
    last=matches[-1]
    if any(c.isalnum() or c in '-‐‑' for c in clean[last.end():]):
        return None,'unsupported_mixed_or_nonword_ending'
    word=last.group()
    return (None,'unsupported_compound_first_endword') if '-' in word else (word,None)


def stressed_vowels(word,positions):
    vowels=[c for c in word if c in 'аеёиоуыэюя']
    return {vowels[p-1] for p in positions if 0<p<=len(vowels)}


def select_case(case,entries,compare,max_choices=32):
    if type(max_choices)is not int or not 1<=max_choices<=32:raise ValueError('Pool bound1..32 required')
    first,reason=first_endword(case['first_line'])
    row=dict(case,case_id=case['id'],first_endword=first,status='unsupported_first_endword' if reason else 'no_candidates',
        failure_reason=reason,candidates=[],checked_frequency_ranked_words=0,phonetic_comparisons=0,
        skipped_compounds=0,skipped_identical_or_folded_duplicates=0,phonetic_status_counts={})
    if reason:return row
    seen={fold(first)};statuses=Counter()
    for index,entry in enumerate(entries):
        row['checked_frequency_ranked_words']+=1
        word=entry['word']
        if '-' in word:
            row['skipped_compounds']+=1;continue
        if WORD.fullmatch(word) is None:raise ValueError('Invalid frozen vocabulary word')
        key=fold(word)
        if key in seen:
            row['skipped_identical_or_folded_duplicates']+=1;continue
        comparison=compare(case['first_line'],word)
        row['phonetic_comparisons']+=1;status=comparison.get('status','unknown');statuses[status]+=1
        if comparison.get('words')!=[first,word]:raise ValueError('Phonetic API lost a full lexical word')
        if status=='error':
            row.update(status='phonetic_error',failure_reason=comparison.get('diagnostic','phonetic_error'));break
        if status=='unknown' and comparison.get('reason')=='missing_stress' and not comparison['stresses'][0][0]:
            row.update(status='unknown_first_stress',failure_reason='missing_first_endword_stress');break
        if status!='ok' or comparison.get('exact_all') is not True:continue
        stresses=comparison['stress_evidence']
        lv=stressed_vowels(first,stresses[0][0]);rv=stressed_vowels(word,stresses[1][0])
        flags=['rpst_stressed_i_y_convention'] if ('и' in lv and 'ы' in rv) or ('ы' in lv and 'и' in rv) else []
        row['candidates'].append(dict(word=word,freq=entry['freq'],workcount=entry['workcount'],
            familycount=entry['familycount'],source_ids=entry['example_work_ids'],
            source_ids_role='up_to5_example_full_train_work_ids',frequency_rank=entry['frequency_rank'],
            source_vocab_index=index,phonetic=comparison,phonetic_policy_flags=flags))
        seen.add(key)  # Preserve frozen v3: dedup only among accepted exact candidates.
        if len(row['candidates'])==max_choices:break
    row['phonetic_status_counts']=dict(statuses)
    if row['status'] in ('phonetic_error','unknown_first_stress'):
        row['candidates']=[]
    elif row['candidates']:row.update(status='ok',failure_reason=None)
    else:row['failure_reason']='no_exact_all_nonidentical_train_word'
    return row


def load_phonetics():
    from scripts.probe_rpst08 import load_verified_tool
    from scripts.rhyme_endings16 import compare_endings
    tool,torch,version=load_verified_tool(ARTIFACTS)
    def compare(left,right):
        with torch.inference_mode():return compare_endings(tool,left,right)
    return compare,dict(rpst_revision=version['rpst_revision'],rpst_version=version['rpst_version'],device='cpu')


def prepare(input_path,output,authorization_path=None,*,phonetics_factory=load_phonetics):
    input_path,output=map(local,(input_path,output))
    if authorization_path is not None:authorization_path=local(authorization_path)
    partial=output.with_suffix('.incomplete.partial.json')
    with run_lock(output,'runtime_pool'):
        if any(p.exists() for p in (output,partial,partial.with_suffix(partial.suffix+'.tmp'))):
            raise FileExistsError('Preserve existing output/partial; no resume')
        frozen={input_path:sha256(input_path),TOKENIZER:TOKENIZER_SHA,VOCAB:VOCAB_SHA}
        def verify():
            for path,digest in frozen.items():
                if sha256(path)!=digest:raise ValueError('Runtime pool source changed: '+str(path))
        verify();payload=read_json(input_path)
        if payload['schema']!='rhyme17-visible-runtime-input-1' or payload['tokenizer_sha256']!=TOKENIZER_SHA:
            raise ValueError('Visible runtime schema/tokenizer mismatch')
        opening=opening_metadata(payload,frozen[input_path],authorization_path)
        if opening:frozen[authorization_path]=opening['sha256']
        cases=project_cases(payload,Tokenizer.from_file(str(TOKENIZER)))
        vocab=read_json(VOCAB);entries=vocab['entries']
        if vocab['split']!='train' or vocab['records']!=5478 or len(entries)!=85912:
            raise ValueError('Frozen full poetry train vocabulary required')
        if entries!=sorted(entries,key=lambda r:(-r['freq'],r['word'])):
            raise ValueError('Vocabulary frequency order changed')
        # The authenticated vocabulary is sufficient at inference. Its build
        # provenance is retained below; training corpora are not runtime inputs.
        for name in SOURCES:frozen[ROOT/name]=sha256(ROOT/name)
        version_path=ARTIFACTS/'version.json';frozen[version_path]=sha256(version_path)
        version=read_json(version_path)
        if version['rpst_revision']!=RPST_REVISION:raise ValueError('Frozen RPST revision required')
        for item in version['verified_files']:
            path=(ARTIFACTS/item['path']).resolve()
            if not path.is_relative_to(ARTIFACTS.resolve()):raise ValueError('RPST path escape')
            frozen[path]=item['sha256']
        verify()
        exclusions=[dict(word=e['word'],source_vocab_index=i,frequency_rank=e['frequency_rank'])
            for i,e in enumerate(entries) if '-' in e['word']]
        report=dict(schema='rhyme17-runtime-pool-1',complete=False,training=False,gpu=False,
            split=payload['split'],test_used=payload['test_used'],input_sha256=frozen[input_path],
            tokenizer_sha256=TOKENIZER_SHA,vocabulary_sha256=VOCAB_SHA,opening_authorization=opening,
            case_projection_fields=list(FIELDS),case_selection_fields=['id','first_line'],
            gold_ending_or_reference_used=False,results=[],
            policy=dict(max_choices=32,frequency_descending=True,phonetic='frozen RPST exact_all',
                nonhyphen_only=True,exact_all_required=True,yo_e_dedup=True,
                compounds='excluded before phonetic comparison; unsupported first-word compounds retained as failures',
                dedup='casefold yo->e among accepted exact candidates; first accepted frequency-ranked spelling retained (frozen v3)',
                stressed_i_y='frozen convention flagged, not corrected',
                vocabulary_scope='all85912 wordforms from5478 train poetry works; no validation/test sources consulted'),
            unsupported_compound_exclusion_count=len(exclusions),unsupported_compound_exclusions=exclusions,
            sources_sha256={relative(p):digest for p,digest in frozen.items()},
            vocabulary_build_sources_sha256=vocab['sources_sha256'],
            limitation='Phonetic pool is not semantic quality; authorization metadata is an audit assertion, not proof of access restrictions. No original sealed file is needed by this API.')
        output.parent.mkdir(parents=True,exist_ok=True);write_json(partial,report)
        compare,evidence=phonetics_factory();report['rpst']=evidence
        if evidence['rpst_revision']!=version['rpst_revision']:raise ValueError('Loaded RPST revision mismatch')
        for case in cases:
            report['results'].append(select_case(case,entries,compare));write_json(partial,report)
        if [r['id'] for r in report['results']]!=[c['id'] for c in cases]:raise ValueError('Case coverage mismatch')
        verify();report.update(complete=True,cases=len(cases),status_counts=dict(Counter(r['status'] for r in report['results'])),
            input_pre_post_sha_equal=True)
        write_json(partial,report)
        if output.exists():raise FileExistsError(output)
        partial.rename(output)
    return dict(output=relative(output),sha256=sha256(output),cases=len(cases),status_counts=report['status_counts'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    print(prepare(args.input,args.output,args.authorization))

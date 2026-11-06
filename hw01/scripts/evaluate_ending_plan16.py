"""Free ending-plan generation and CPU scoring; no reference-ending input.

Modes are reported separately: planned formatting versus normal continuation.
The full raw output is retained. No retries, oracle prefix or rhyme-based selection.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poetrygpt.data import read_json, sha256
from scripts.prepare_ending_plan16 import SERIALIZATION, PLAN_MARKER, LINE_MARKER, write_json
from scripts.rhyme_endings16 import last_word, compare_endings


def load_cases(folder, tokenizer):
    folder = Path(folder)
    manifest = read_json(folder/'manifest.json')
    if manifest.get('serialization') != SERIALIZATION or manifest['evaluation']['oracle'] is not False:
        raise ValueError('Expected non-oracle ending-plan dataset')
    if manifest['protocol']['plan_marker'] != PLAN_MARKER or manifest['protocol']['line_marker'] != LINE_MARKER:
        raise ValueError('Changed generation protocol')
    path = folder/'dev_prefixes.json'
    if sha256(path) != manifest['output_sha256'][path.name]:
        raise ValueError('Changed dev prefix panel')
    panel = read_json(path)
    if set(panel) != {'schema', 'source_manifest_sha256', 'cases'} or panel['schema'] != 'ending-plan16-visible-prefixes-1':
        raise ValueError('Unexpected panel schema')
    if panel['source_manifest_sha256'] != manifest['source_manifest_sha256']:
        raise ValueError('Dev/source mismatch')
    cases = panel['cases']
    if (not cases or len({c['id'] for c in cases}) != len(cases)
            or len(cases) != manifest['validation']['records']):
        raise ValueError('Empty or duplicate dev cases')
    for case in cases:
        if set(case) != {'id','group_id','context_ids','context_text','first_line'}:
            raise ValueError('Dev must contain visible prefixes only; reference/assistant prefix forbidden')
        ids = case['context_ids']
        if not ids or any(type(i) is not int for i in ids):
            raise ValueError('Invalid visible token prefix')
        text = tokenizer.decode(ids, skip_special_tokens=False)
        visible = tokenizer.decode(ids, skip_special_tokens=True)
        lines = [line for line in visible.splitlines() if line.strip()]
        if (text != case['context_text'] or not text.endswith('\n') or not lines
                or lines[-1] != case['first_line']
                or tokenizer.encode(text, add_special_tokens=False).ids != ids):
            raise ValueError('Visible prefix roundtrip/line mismatch')
    return manifest, cases


def prompt_ids(case, mode, tokenizer):
    if mode == 'normal':
        return list(case['context_ids'])
    if mode != 'planned':
        raise ValueError('Unknown generation mode')
    ids = tokenizer.encode(case['context_text'] + PLAN_MARKER, add_special_tokens=False).ids
    if ids[:len(case['context_ids'])] != case['context_ids']:
        raise ValueError('Prompt retokenized visible context')
    return ids


def select_panel(cases, panel, tokenizer):
    """Match only existing visible val contexts; adopt the old EOS input convention."""
    from poetrygpt.generate import encode_continuation
    by_text = {}
    for case in cases:
        text = tokenizer.decode(case['context_ids'],skip_special_tokens=True)
        if text in by_text:
            raise ValueError('Ambiguous visible validation prefix')
        by_text[text] = case
    allowed = {'id','family_id','author','kind','prompt','max_new_tokens','temperature','top_k','top_p','seeds'}
    selected, mapping = [], []
    if panel.get('split') != 'validation' or panel.get('test_used') is not False:
        raise ValueError('Only declared development validation panels allowed')
    for row in panel['cases']:
        if set(row) != allowed or row['kind'] != 'continuation':
            raise ValueError('Panel contains unsupported fields; references/options/prefixes forbidden')
        source = by_text.get(row['prompt'])
        if source is None or row['family_id'] != source['group_id']:
            raise ValueError('Panel is not an exact visible validation subset')
        ids = encode_continuation(tokenizer,row['prompt'])
        selected.append(dict(source,id=row['id'],context_ids=ids,
                             context_text=tokenizer.decode(ids,skip_special_tokens=False)))
        mapping.append(dict(panel_id=row['id'],source_id=source['id'],
                            prefix_ids_changed=ids!=source['context_ids']))
    if not selected or len({c['id'] for c in selected}) != len(selected):
        raise ValueError('Empty or duplicate panel selection')
    return selected,mapping


def parse_output(text, mode, stop_reason):
    plan, body = None, text
    if mode == 'planned':
        match = re.match(r'([А-Яа-яЁё]+)' + re.escape(LINE_MARKER), text)
        if not match:
            return dict(format='malformed_plan', complete=False, plan=None, line=None, compliance=False)
        plan, body = match.group(1), text[match.end():]
    elif mode != 'normal':
        raise ValueError('Unknown generation mode')
    line, separator, tail = body.partition('\n')
    complete = bool(separator) or stop_reason == 'stop_token'
    if not line.strip() or last_word(line) is None:
        status = 'empty_or_non_cyrillic_line'
    else:
        status = 'complete' if complete else 'incomplete_line'
    compliance = (mode == 'planned' and status == 'complete'
                  and last_word(line).replace('ё','е') == plan.lower().replace('ё','е'))
    return dict(format=status, complete=status == 'complete', plan=plan, line=line,
                compliance=bool(compliance), trailing_characters=len(tail))


def token_budget(ids, text, mode, stop_reason, tokenizer, body_limit, plan_limit):
    """Count actual sampled IDs, including any token spanning both text regions."""
    parsed = parse_output(text,mode,stop_reason)
    if not parsed['complete']:
        return dict(eligible=False,reason=parsed['format'],plan_tokens=None,body_tokens=None,
                    body_limit=body_limit,plan_limit=plan_limit)
    plan_text = (parsed['plan']+LINE_MARKER) if mode=='planned' else ''
    complete_prefix = plan_text+parsed['line']
    if text.startswith(complete_prefix+'\n'):
        complete_prefix += '\n'
    else:
        # EOS/EOT makes the final line complete; its stop token consumes budget.
        complete_prefix = None
    decoded = [tokenizer.decode(ids[:n],skip_special_tokens=True) for n in range(len(ids)+1)]
    plan_end = next(n for n,prefix in enumerate(decoded) if prefix.startswith(plan_text))
    body_end = len(ids) if complete_prefix is None else next(
        n for n,prefix in enumerate(decoded) if prefix.startswith(complete_prefix))
    shared = int(mode=='planned' and len(decoded[plan_end])>len(plan_text))
    body_tokens = body_end-plan_end+shared
    eligible = body_tokens<=body_limit and plan_end<=plan_limit
    return dict(eligible=eligible,reason='within_budget' if eligible else 'region_budget_exceeded',
                plan_tokens=plan_end,body_tokens=body_tokens,body_limit=body_limit,plan_limit=plan_limit)


def score_samples(samples, rhyme):
    results, totals = [], {}
    seen = set()
    for sample in samples:
        key = (sample['mode'], sample['case']['id'], sample['seed'])
        if key in seen:
            raise ValueError('Duplicate mode/case/seed result')
        seen.add(key)
        parsed = parse_output(sample['text'], sample['mode'], sample['stop_reason'])
        budget = sample['token_budget']
        # Compare actual generated body with the visible source line, never the plan.
        sound = rhyme(sample['case']['first_line'], parsed['line']) if parsed['complete'] and budget['eligible'] else None
        results.append(dict(mode=sample['mode'], case_id=sample['case']['id'], seed=sample['seed'],
                            **parsed, token_budget=budget, rhyme=sound))
    for mode in sorted({r['mode'] for r in results}):
        rows = [r for r in results if r['mode'] == mode]
        counts = dict(samples=len(rows), complete_lines=sum(r['complete'] for r in rows),
            budget_eligible_lines=sum(r['complete'] and r['token_budget']['eligible'] for r in rows),
            format_counts=dict(Counter(r['format'] for r in rows)),
            plan_compliance=sum(r['compliance'] and r['token_budget']['eligible'] for r in rows) if mode == 'planned' else None,
            rhyme_statuses=dict(Counter((r['rhyme'] or {}).get('status','no_complete_line') for r in rows)),
            exact_any=sum(bool((r['rhyme'] or {}).get('exact_any')) for r in rows),
            exact_all=sum(bool((r['rhyme'] or {}).get('exact_all')) for r in rows))
        counts['exact_all_per_attempt'] = counts['exact_all']/len(rows)
        counts['exact_any_per_attempt'] = counts['exact_any']/len(rows)
        totals[mode] = counts
    return dict(summary=totals, results=results,
        limitations='No semantic/style score. Exact-any may choose the wrong lexical stress. '
                    'Incomplete or region-budget-exceeding lines remain in per-attempt denominators. '
                    'Planned and normal modes are different protocols; oracle results are unsupported.')


def generate_report(args):
    import torch
    from tokenizers import Tokenizer
    from poetrygpt.generate import generate
    from poetrygpt.model import GPT, ModelConfig
    from poetrygpt.runtime import load_checkpoint, run_lock, seed_everything
    from poetrygpt.serialization import special_ids
    from scripts import shared_gpu11 as shared
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    manifest, cases = load_cases(args.data, tokenizer)
    selection = None
    if args.panel is not None:
        panel = read_json(args.panel)
        cases,mapping = select_panel(cases,panel,tokenizer)
        for row in panel['cases']:
            if any(row[key] != getattr(args,key) for key in ('seeds','max_new_tokens','temperature','top_k','top_p')):
                raise ValueError('Normal-control sampling parameters must match the selected panel')
        selection = dict(panel=str(args.panel),sha256=sha256(args.panel),mapping=mapping,
            prefix_policy='Identical old encode_continuation EOS prefix for both modes; training contexts unchanged')
    token_hash = sha256(args.tokenizer)
    if token_hash not in [v for k,v in manifest['inputs_sha256'].items() if k.replace('\\','/').endswith('tokenizer.json')]:
        raise ValueError('Dataset tokenizer mismatch')
    hashes = dict(checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.data/'manifest.json'),
                  tokenizer_sha256=token_hash, panel_sha256=sha256(args.data/'dev_prefixes.json'))
    with run_lock(ROOT/'runs/pretrain_series01', 'series'):
        if args.device == 'cuda':
            shared.AUDIT = ROOT/'audit/poetry16/gpu_admission.jsonl'
            shared.admit(torch, 3072, shared.Audit(phase='ending_plan16_generation', evaluator_sha256=sha256(__file__)))
        torch.set_num_threads(4)
        state = load_checkpoint(args.checkpoint)
        if state['tokenizer_sha256'] != token_hash:
            raise ValueError('Checkpoint tokenizer mismatch')
        model = GPT(ModelConfig(**state['model_config']))
        model.load_state_dict(state['model'])
        model.to(args.device).eval()
        del state
        specials = special_ids(tokenizer)
        stops = [specials[s] for s in ('<|eos|>','<|eot|>')]
        forbidden = [specials[s] for s in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>')]
        report = dict(schema='ending-plan16-free-generation-1', oracle=False,
            checkpoint=str(args.checkpoint), data=str(args.data), model=model.config.__dict__,
            evaluator_sha256=sha256(__file__), protocol_sha256=sha256(ROOT/'scripts/prepare_ending_plan16.py'),
            modes=args.modes, seeds=args.seeds, case_ids=[c['id'] for c in cases], max_new_tokens=args.max_new_tokens,
            planned_extra_tokens=args.plan_extra_tokens, selection=selection,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            samples=[], **hashes)
        for case in cases:
            for mode in args.modes:
                prompt = prompt_ids(case, mode, tokenizer)
                budget = args.max_new_tokens+(args.plan_extra_tokens if mode=='planned' else 0)
                if len(prompt)+budget > model.config.context_length:
                    raise ValueError('Generation would truncate visible context')
                for seed in args.seeds:
                    seed_everything(seed+int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16))
                    ids, stop = generate(model,prompt,max_new_tokens=budget,
                        temperature=args.temperature,top_k=args.top_k,top_p=args.top_p,
                        stop_ids=stops,forbidden_ids=forbidden,precision='bf16' if args.device=='cuda' else 'fp32')
                    text = tokenizer.decode(ids,skip_special_tokens=True)
                    regions = token_budget(ids,text,mode,stop,tokenizer,args.max_new_tokens,
                                           args.plan_extra_tokens if mode=='planned' else 0)
                    report['samples'].append(dict(case=case,mode=mode,seed=seed,
                        text=text,generated_ids=ids,token_budget=regions,
                        generated_tokens=len(ids),prompt_tokens=len(prompt),max_new_tokens=budget,stop_reason=stop))
        if (sha256(args.checkpoint) != hashes['checkpoint_sha256']
                or sha256(args.data/'manifest.json') != hashes['manifest_sha256']
                or sha256(args.data/'dev_prefixes.json') != hashes['panel_sha256']
                or (selection is not None and sha256(args.panel) != selection['sha256'])):
            raise ValueError('Evaluation input changed')
        write_json(args.output,report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    generation = commands.add_parser('generate')
    generation.add_argument('--data',type=Path,required=True)
    generation.add_argument('--checkpoint',type=Path,required=True)
    generation.add_argument('--tokenizer',type=Path,required=True)
    generation.add_argument('--output',type=Path,required=True)
    generation.add_argument('--panel',type=Path,help='Optional exact visible subset using old continuation EOS convention')
    generation.add_argument('--device',choices=('cpu','cuda'),required=True)
    generation.add_argument('--modes',choices=('planned','normal'),nargs='+',default=['planned','normal'])
    generation.add_argument('--seeds',type=int,nargs='+',default=[7,19])
    generation.add_argument('--max-new-tokens',type=int,default=24)
    generation.add_argument('--plan-extra-tokens',type=int,default=16)
    generation.add_argument('--temperature',type=float,default=.7)
    generation.add_argument('--top-k',type=int,default=40)
    generation.add_argument('--top-p',type=float,default=.9)
    scoring = commands.add_parser('score')
    scoring.add_argument('--input',type=Path,required=True)
    scoring.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'generate':
        if args.max_new_tokens < 1 or args.plan_extra_tokens < 0:
            raise ValueError('Invalid per-mode generation budgets')
        if len(set(args.modes)) != len(args.modes) or len(set(args.seeds)) != len(args.seeds):
            raise ValueError('Duplicate modes or seeds')
        generate_report(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        source = read_json(args.input)
        if source.get('oracle') is not False or source.get('schema') != 'ending-plan16-free-generation-1':
            raise ValueError('Only own free-generation reports accepted')
        expected = {(mode,case_id,seed) for case_id in source['case_ids']
                    for mode in source['modes'] for seed in source['seeds']}
        actual = {(r['mode'],r['case']['id'],r['seed']) for r in source['samples']}
        if actual != expected:
            raise ValueError('Incomplete mode/seed coverage')
        from scripts.probe_rpst08 import load_verified_tool, DEFAULT_ARTIFACTS
        tool,torch,version = load_verified_tool(DEFAULT_ARTIFACTS)
        with torch.inference_mode():
            report = score_samples(source['samples'],lambda left,right: compare_endings(tool,left,right))
        write_json(args.output,dict(schema='ending-plan16-free-scores-1',oracle=False,
            source_sha256=sha256(args.input),checkpoint_sha256=source['checkpoint_sha256'],
            evaluator_sha256=sha256(__file__),rhyme_helper_sha256=sha256(ROOT/'scripts/rhyme_endings16.py'),
            rpst_revision=version['rpst_revision'],**report))


if __name__ == '__main__':
    main()

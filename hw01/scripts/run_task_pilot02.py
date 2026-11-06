"""Bounded task diagnostics: memorization, paired answers, and one-line continuation."""
import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.generate import encode_continuation, generate
from poetrygpt.model import GPT, ModelConfig
from poetrygpt.runtime import (code_hashes, environment, load_checkpoint, resolve_device,
                              run_lock, seed_everything, write_json)
from poetrygpt.train import train, validated_config
from scripts.training_data import encode_prompt, special_ids, SPECIAL_TOKENS

TOKENIZER = ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'
TRAIN_LIMITS = {'memorization32': 512, 'paired_prose': 250, 'paired_verse': 250, 'continuation': 250}
PARENTS = ('runs/pretrain_series01/r08_66m_cooldown/last.pt',
           'runs/pilot01/m66_poetry/mid.pt', 'runs/pilot01/m66_poetry/last.pt')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def project_path(path):
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError('Path must stay inside this project')
    return result


def validate_train_config(raw, parent_config, name, checkpoint):
    if name not in TRAIN_LIMITS:
        raise ValueError('Only the four named task diagnostics may train')
    if project_path(checkpoint) not in {project_path(p) for p in PARENTS}:
        raise ValueError('Training requires an approved own-model parent')
    cfg = validated_config(raw)
    if cfg['domain'] != 'sft' or cfg['max_steps'] > TRAIN_LIMITS[name]:
        raise ValueError('Only bounded SFT-container training is authorized')
    if cfg['model'] != ModelConfig(**parent_config).__dict__:
        raise ValueError('The parent architecture must remain unchanged')
    if (cfg['batch_size'] > 8 or cfg['accumulation_steps'] != 1
            or cfg['cpu_threads'] != 4 or cfg['precision'] != 'bf16'):
        raise ValueError('Require BF16, CPU 4, batch <= 8, accumulation 1')
    if project_path(cfg['data']) != project_path(f'data/prepared_pilot02/{name}'):
        raise ValueError('Each diagnostic must use its own prepared_pilot02 data')
    if project_path(cfg['tokenizer']) != TOKENIZER.resolve():
        raise ValueError('Selected tokenizer must remain unchanged')
    return cfg


def validate_panel(panel):
    if not isinstance(panel, dict) or panel.get('seeds') != [7, 2026]:
        raise ValueError('Panel must freeze seeds [7, 2026]')
    cases = panel.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError('Panel must contain 1..100 cases')
    identifiers = set()
    for case in cases:
        identifier = case.get('id')
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError('Each case needs a unique nonempty ID')
        identifiers.add(identifier)
        if case.get('kind') == 'raw':
            prefix = case.get('prefix')
            if not isinstance(prefix, str) or not prefix.strip() or '\n' in prefix or '\r' in prefix:
                raise ValueError('Raw prefix must be exactly one line, without a trailing newline')
            contents = [prefix]
        elif case.get('kind') == 'chat':
            messages = case.get('messages')
            if not isinstance(messages, list) or len(messages) != 1:
                raise ValueError('These task diagnostics require exactly one user message')
            if messages[0].get('role') != 'user':
                raise ValueError('Reference assistant answers cannot enter the prompt')
            contents = [messages[0].get('content')]
        else:
            raise ValueError('Case kind must be chat or raw')
        if any(not isinstance(text, str) or not text.strip()
               or any(marker in text for marker in SPECIAL_TOKENS) for text in contents):
            raise ValueError('Prompt content must be nonempty and contain no special markers')
    return cases


def encode_case(tokenizer, case, context_length, budget):
    if case['kind'] == 'raw':
        prompt = encode_continuation(tokenizer, case['prefix'] + '\n')
    else:
        prompt = encode_prompt(tokenizer, case['messages'])
    if len(prompt) + budget > context_length:
        raise ValueError(f"Case {case['id']} cannot fit its full generation budget; no truncation")
    return prompt


def require_resources(training):
    """Fail before allocating model weights; never stop or modify other jobs."""
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys', 'total_page',
                'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    ram_free = memory.avail_phys / 1024**3
    disk_free = shutil.disk_usage(ROOT).free / 1024**3
    if ram_free < (6 if training else 4) or disk_free < (4 if training else 1):
        raise RuntimeError(f'Insufficient free host resources: RAM {ram_free:.2f}, disk {disk_free:.2f} GiB')
    device, precision = resolve_device('cuda', 'bf16')
    free, total = torch.cuda.mem_get_info(device)
    if free / 1024**3 < (10 if training else 3):
        raise RuntimeError(f'Insufficient free GPU memory: {free / 1024**3:.2f} GiB; no process stopped')
    return device, precision, {'ram_free_gib': ram_free, 'disk_free_gib': disk_free,
                               'gpu_free_gib': free / 1024**3, 'gpu_total_gib': total / 1024**3}


def training_pilot(args, report, output):
    raw = read_json(args.config)
    state = load_checkpoint(args.checkpoint)
    cfg = validate_train_config(raw, state['model_config'], args.name, args.checkpoint)
    if state['tokenizer_sha256'] != sha256(TOKENIZER):
        raise ValueError('Parent tokenizer differs from selected tokenizer')
    data = PreparedDataset(project_path(cfg['data']), 'sft', 'train', cfg['model']['context_length'])
    if args.name == 'memorization32' and len(data) != 32:
        raise ValueError('Memorization diagnostic must contain exactly 32 train records')
    report.update(config=cfg, train_records=len(data), data_manifest_sha256=sha256(data.folder / 'manifest.json'),
                  serialization=data.manifest.get('serialization', 'chat_template'),
                  optimizer_policy='Fresh optimizer; own checkpoint weights only')
    del state, data
    gc.collect()
    write_json(output, report)
    run_dir = ROOT / 'runs/pilot02' / args.name
    train(cfg, run_dir, device_name='cuda', init_from=args.checkpoint)
    status = read_json(run_dir / 'status.json')
    if status['status'] not in ('completed', 'early_stopped'):
        raise RuntimeError(f'Diagnostic did not finish normally: {status}')
    report.update(training_status=status, training_performed=True,
                  validation=read_json(run_dir / 'validation_latest.json'),
                  checkpoints={name: {'path': str(run_dir / name), 'sha256': sha256(run_dir / name)}
                               for name in ('last.pt', 'best.pt') if (run_dir / name).exists()})


def generate_panel(args, report, output, device, precision):
    panel = read_json(args.panel)
    cases = validate_panel(panel)
    state = load_checkpoint(args.checkpoint)
    if state['tokenizer_sha256'] != sha256(TOKENIZER):
        raise ValueError('Checkpoint tokenizer differs from selected tokenizer')
    model = GPT(ModelConfig(**state['model_config']))
    model.load_state_dict(state['model'])
    report.update(model_config=state['model_config'], checkpoint_step=state['step'], lineage=state['lineage'])
    del state
    gc.collect()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    special = special_ids(tokenizer)
    prompts = [encode_case(tokenizer, case, model.config.context_length, args.max_new_tokens) for case in cases]
    model.to(device)
    stops = [special[t] for t in ('<|eos|>', '<|eot|>')]
    forbidden = [special[t] for t in ('<|pad|>', '<|bos|>', '<|user|>', '<|assistant|>')]
    report.update(training_performed=False, samples=[],
                  settings={'temperature': .7, 'top_k': 40, 'top_p': .9, 'seeds': panel['seeds'],
                            'max_new_tokens': args.max_new_tokens, 'precision': precision,
                            'newline_is_stop': False},
                  serialization={'raw': "EOS + encode(prefix + '\\n')", 'chat': 'existing encode_prompt',
                                 'first_nonempty_line': 'Display field only; full generation retained'})
    for case, prompt in zip(cases, prompts):
        for base_seed in panel['seeds']:
            seed = base_seed + int(hashlib.sha256(case['id'].encode()).hexdigest()[:8], 16)
            seed_everything(seed)
            started = time.perf_counter()
            ids, reason = generate(model, prompt, max_new_tokens=args.max_new_tokens,
                                   temperature=.7, top_k=40, top_p=.9,
                                   stop_ids=stops, forbidden_ids=forbidden, precision=precision)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            report['samples'].append({'id': case['id'], 'kind': case['kind'], 'case': deepcopy(case),
                'base_seed': base_seed, 'seed': seed, 'text': text, 'full_text': text,
                'first_nonempty_line': next((line.strip() for line in text.splitlines() if line.strip()), ''),
                'prompt_token_ids': prompt, 'prompt_tokens': len(prompt), 'token_ids': ids,
                'generated_tokens': len(ids), 'stop_reason': reason,
                'stop_token_id': ids[-1] if ids and ids[-1] in stops else None,
                'seconds': time.perf_counter() - started})
            write_json(output, report)
    torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'generate'), required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=48)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', args.name):
        parser.error('Name must be a simple unique identifier')
    if ((args.mode == 'train') != bool(args.config) or (args.mode == 'generate') != bool(args.panel)
            or not 1 <= args.max_new_tokens <= 48):
        parser.error('Use config for train, panel for generate, and 1..48 generated tokens')
    args.checkpoint = project_path(args.checkpoint)
    output = ROOT / 'audit/pilot02' / f'{args.name}.json'
    run_dir = ROOT / 'runs/pilot02' / args.name
    torch.set_num_threads(4)
    with run_lock(ROOT / 'runs/pretrain_series01', 'series'):
        if output.exists() or (args.mode == 'train' and run_dir.exists()):
            raise FileExistsError('Diagnostic name already used; refusing overwrite or hidden rerun')
        if args.mode == 'train':
            if args.name not in TRAIN_LIMITS or args.checkpoint not in {project_path(p) for p in PARENTS}:
                raise ValueError('Unapproved training name or parent')
            args.config = project_path(args.config)
        else:
            args.panel = project_path(args.panel)
            validate_panel(read_json(args.panel))
        device, precision, resources = require_resources(args.mode == 'train')
        inputs = {'checkpoint': sha256(args.checkpoint), 'runner': sha256(Path(__file__)),
                  'code': code_hashes(), 'tokenizer': sha256(TOKENIZER),
                  'config' if args.mode == 'train' else 'panel': sha256(args.config or args.panel)}
        torch.cuda.reset_peak_memory_stats(device)
        report = {'name': args.name, 'mode': args.mode, 'status': 'running',
                  'checkpoint': str(args.checkpoint), 'inputs_sha256': inputs, 'started_at': utc_now(),
                  'environment': environment(device, precision), 'resource_preflight': resources}
        write_json(output, report)
        started = time.perf_counter()
        try:
            if args.mode == 'train':
                training_pilot(args, report, output)
            else:
                generate_panel(args, report, output, device, precision)
            report['status'] = 'completed'
        except BaseException as error:
            report.update(status='failed_or_interrupted', error=repr(error))
            raise
        finally:
            report.update(finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started,
                          peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                          peak_cuda_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3)
            write_json(output, report)
        print(f'{args.name}: completed; report={output}', flush=True)


if __name__ == '__main__':
    main()

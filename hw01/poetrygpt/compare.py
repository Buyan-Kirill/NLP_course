"""Fixed validation-only comparison of own stage checkpoints and free-running replies."""
import argparse
import gc
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from .serialization import encode_prompt
from .data import PreparedDataset, read_json, sha256
from .evaluate import evaluate
from .generate import encode_continuation, generate
from .model import GPT, ModelConfig
from .runtime import load_checkpoint, resolve_device, seed_everything, write_json


def sample_case(model, tokenizer, case, precision):
    sample = {'id': case['id'], 'kind': case['kind'], 'turns': []}
    history = []
    for user in case.get('user_turns', [None]):
        if case['kind'] == 'continuation':
            prompt = encode_continuation(tokenizer, case['prefix'])
        else:
            history.append({'role': 'user', 'content': user})
            prompt = encode_prompt(tokenizer, history)
        if len(prompt) >= model.config.context_length:
            sample['turns'].append({'user': user, 'text': '', 'stop_reason': 'prompt_context_overflow'})
            break
        ids, reason = generate(model, prompt, max_new_tokens=80, temperature=.8, top_k=40,
                               stop_ids=(2, 5), forbidden_ids=(0, 1, 3, 4), precision=precision)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        sample['turns'].append({'user': user, 'text': text, 'stop_reason': reason, 'tokens': len(ids)})
        if case['kind'] == 'chat':
            if not text.strip():
                sample['dialogue_terminated_reason'] = 'empty_model_response'
                break
            # Evaluate the actual generated dialogue, not a gold first answer.
            history.append({'role': 'assistant', 'content': text})
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, default=Path('configs/evaluation_v4.json'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    suite = read_json(args.prompts)
    if suite['split'] != 'validation' or sha256(suite['source']) != suite['source_sha256']:
        raise ValueError('Fixed validation prompts/source changed')
    output = args.run_dir / 'comparison'
    if output.exists():
        raise ValueError('Refuse to overwrite an earlier comparison')
    started = time.perf_counter()
    report = {'status': 'running', 'prompt_sha256': sha256(args.prompts),
              'raw_continuation_boundary': 'eos',
              'test_used': False, 'automatic_literary_scoring': False, 'stages': {}}
    for stage in ('prose', 'poetry', 'sft'):
        checkpoint = args.run_dir / stage / 'best.pt'
        state = load_checkpoint(checkpoint)
        cfg = state['config']
        device, precision = resolve_device(args.device, cfg['precision'])
        torch.set_num_threads(cfg['cpu_threads'])
        model = GPT(ModelConfig(**state['model_config'])).to(device)
        model.load_state_dict(state['model'])
        tokenizer = Tokenizer.from_file(cfg['tokenizer'])
        if sha256(cfg['tokenizer']) != state['tokenizer_sha256']:
            raise ValueError('Tokenizer mismatch')
        if sha256(Path(cfg['data']) / 'manifest.json') != state['prepared_manifest_sha256']:
            raise ValueError('Evaluation data release differs from checkpoint')
        result = {'checkpoint_sha256': sha256(checkpoint), 'step': state['step'],
                  'validation': {}, 'samples': []}
        for domain in ('prose', 'poetry', 'sft'):
            data = PreparedDataset(cfg['data'], domain, 'validation', cfg['model']['context_length'])
            evaluation = evaluate(model, data, device, precision, cfg['batch_size'])
            write_json(output / f'{stage}_on_{domain}.json', evaluation)
            result['validation'][domain] = evaluation['metrics']['overall']
            del data
        for index, case in enumerate(suite['cases']):
            seed_everything(42 + index)
            result['samples'].append(sample_case(model, tokenizer, case, precision))
        report['stages'][stage] = result
        write_json(output / 'report.json', report)
        del state, model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report.update(status='completed', seconds=time.perf_counter() - started)
    write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()

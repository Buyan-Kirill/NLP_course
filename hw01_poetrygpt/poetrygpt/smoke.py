"""Six GPU updates per stage, resume, full validation and short inference."""
import argparse
import gc
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from .serialization import encode_prompt
from .data import PreparedDataset, read_json, sha256
from .evaluate import evaluate
from .generate import generate
from .model import GPT, ModelConfig
from .runtime import environment, load_checkpoint, seed_everything, write_json
from .train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='Optional prepared release for a functional test before expanded SFT is ready')
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This smoke test requires CUDA and BF16 support; CPU fallback is forbidden')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError('Smoke output directory must be new or empty')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    torch.set_num_threads(4)
    report = {'purpose': 'Functional test, not model-quality evaluation', 'environment': environment(device, 'bf16'),
              'stages': {}, 'status': 'running', 'test_split_used_for_model_evaluation': False}
    write_json(args.run_dir / 'summary.json', report)
    parent = None
    started = time.perf_counter()
    for domain in ('prose', 'poetry', 'sft'):
        cfg = read_json(f'configs/training/{domain}_17m.json')
        if args.data is not None:
            cfg['data'] = str(args.data)
        cfg.update(max_steps=6, warmup_steps=2, eval_interval=3, save_interval=3,
                   eval_batches=2 if domain != 'sft' else 0, early_stopping_patience=0,
                   precision='bf16', deterministic=False)
        stage_dir = args.run_dir / domain
        stage_started = time.perf_counter()
        if domain == 'prose':
            partial = train(cfg, stage_dir, 'cuda', stop_after=3)
            state = load_checkpoint(partial)
            assert state['step'] == 3 and state['lineage']['origin'] == 'random_initialization'
            del state
            train(cfg, stage_dir, 'cuda', resume=partial)
        else:
            train(cfg, stage_dir, 'cuda', init_from=parent)
        stage_training_seconds = time.perf_counter() - stage_started
        last = load_checkpoint(stage_dir / 'last.pt')
        assert last['step'] == 6 and len(last['lineage']['parents']) == len(report['stages'])
        parent = stage_dir / 'best.pt'
        state = load_checkpoint(parent)
        model = GPT(ModelConfig(**state['model_config'])).to(device)
        model.load_state_dict(state['model'])
        validation = PreparedDataset(cfg.get('data', 'data/prepared_v3'), domain, 'validation')
        evaluated = evaluate(model, validation, device, 'bf16', batch_size=8)
        write_json(stage_dir / 'full_validation.json', evaluated)
        tokenizer_path = Path('artifacts/tokenizers_v3/selected/tokenizer.json')
        assert sha256(tokenizer_path) == state['tokenizer_sha256']
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        prompt = encode_prompt(tokenizer, [{'role': 'user', 'content': 'Что даёт нам солнце?'}]) if domain == 'sft' else [1] + tokenizer.encode('Над тихой рощей').ids
        seed_everything(42)
        ids, reason = generate(model, prompt, max_new_tokens=24, stop_ids=(2, 5), forbidden_ids=(0, 1, 3, 4), precision='bf16')
        report['stages'][domain] = {
            'status': read_json(stage_dir / 'status.json'), 'training_seconds_including_partial_eval_and_io': stage_training_seconds,
            'parameters': model.parameter_count(), 'selected_checkpoint_step': state['step'],
            'selected_checkpoint_sha256': sha256(parent), 'lineage': state['lineage'],
            'full_validation': evaluated['metrics']['overall'],
            'generation_functionality_only': {'text': tokenizer.decode(ids), 'generated_tokens': len(ids), 'stop_reason': reason},
            'peak_allocated_mib_through_validation': torch.cuda.max_memory_allocated() / 1024**2,
            'peak_reserved_mib_through_validation': torch.cuda.max_memory_reserved() / 1024**2,
            'resume_exercised': domain == 'prose', 'bitwise_gpu_resume_claimed': False}
        write_json(args.run_dir / 'summary.json', report)
        del model, last, state, validation
        gc.collect()
        torch.cuda.empty_cache()
    report.update(status='passed', total_seconds=time.perf_counter() - started)
    write_json(args.run_dir / 'summary.json', report)
    print(f"GPU smoke passed; report: {args.run_dir / 'summary.json'}", flush=True)


if __name__ == '__main__':
    main()

"""Run the three full training stages in one independently managed process."""
import argparse
import gc
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .serialization import encode_prompt
from .data import PreparedDataset, read_json, sha256
from .evaluate import evaluate
from .generate import encode_continuation, generate
from .model import GPT, ModelConfig
from .runtime import code_hashes, load_checkpoint, locked_run, resolve_device, seed_everything, write_json
from .train import train, validated_config


@locked_run('root', 'pipeline')
def run_pipeline(root, device_name='cuda', resume=False):
    root = Path(root)
    if resume:
        plan = read_json(root / 'pipeline_plan.json')
        manifests = plan.get('data_manifests', {'data/prepared_v3/manifest.json': plan.get('data_sha256')})
        if plan['code_sha256'] != code_hashes() or any(sha256(p) != h for p, h in manifests.items()):
            raise ValueError('Pipeline code/data changed; inspect before resuming')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('New pipeline directory must be empty')
        stages = {d: validated_config(read_json(f'configs/training/{d}_17m.json')) for d in ('prose', 'poetry', 'sft')}
        manifests = {str(Path(cfg['data']) / 'manifest.json'): sha256(Path(cfg['data']) / 'manifest.json') for cfg in stages.values()}
        plan = {'code_sha256': code_hashes(), 'data_manifests': manifests, 'stages': stages}
        write_json(root / 'pipeline_plan.json', plan)
    report = read_json(root / 'summary.json') if resume and (root / 'summary.json').exists() else {'stages': {}}
    report.update(status='running', test_split_used_for_model_evaluation=False)
    report.pop('error', None)
    write_json(root / 'summary.json', report)
    parent = None
    try:
        for domain in ('prose', 'poetry', 'sft'):
            stage = root / domain
            if (root / 'STOP').exists():
                report['status'] = 'stopped_between_stages'
                write_json(root / 'summary.json', report)
                return
            cfg = plan['stages'][domain]
            status = read_json(stage / 'status.json') if (stage / 'status.json').exists() else {}
            if status.get('status') not in ('completed', 'early_stopped'):
                checkpoint = stage / 'last.pt'
                train(cfg, stage, device_name, resume=checkpoint if checkpoint.exists() else None,
                      init_from=parent if not checkpoint.exists() else None)
                status = read_json(stage / 'status.json')
                if status['status'] not in ('completed', 'early_stopped'):
                    report['status'] = 'stopped'
                    write_json(root / 'summary.json', report)
                    return
            parent = stage / 'best.pt'
            state = load_checkpoint(parent)
            device, precision = resolve_device(device_name, cfg['precision'])
            model = GPT(ModelConfig(**state['model_config'])).to(device)
            model.load_state_dict(state['model'])
            validation = PreparedDataset(cfg['data'], domain, 'validation', cfg['model']['context_length'])
            full = evaluate(model, validation, device, precision, cfg['batch_size'])
            write_json(stage / 'full_validation.json', full)
            tokenizer = Tokenizer.from_file(cfg['tokenizer'])
            prompt = encode_prompt(tokenizer, [{'role': 'user', 'content': 'Что даёт нам солнце?'}]) if domain == 'sft' else encode_continuation(tokenizer, 'Над тихой рощей')
            seed_everything(42)
            ids, reason = generate(model, prompt, max_new_tokens=80, stop_ids=(2, 5), forbidden_ids=(0, 1, 3, 4), precision=precision)
            report['stages'][domain] = {'status': status, 'selected_checkpoint_step': state['step'],
                'lineage': state['lineage'], 'full_validation': full['metrics']['overall'],
                'generation_sample': {'text': tokenizer.decode(ids), 'stop_reason': reason, 'automatically_scored': False}}
            write_json(root / 'summary.json', report)
            del state, model, validation
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        report['status'] = 'completed'
        write_json(root / 'summary.json', report)
    except BaseException as error:
        report.update(status='failed_or_interrupted', error=repr(error))
        write_json(root / 'summary.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    run_pipeline(args.run_dir, args.device, args.resume)


if __name__ == '__main__':
    main()

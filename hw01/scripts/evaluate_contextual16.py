"""Teacher-forced contextual second-line diagnostic, separate from generation."""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from tokenizers import Tokenizer
from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.evaluate import evaluate
from poetrygpt.model import GPT, ModelConfig
from poetrygpt.runtime import autocast, load_checkpoint, run_lock, write_json
from scripts import shared_gpu11 as shared


def ending_positions(ids, labels, tokenizer):
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if encoded.ids != ids.tolist():
        raise ValueError('Cannot align ending word: source token roundtrip differs')
    words = list(re.finditer('[А-Яа-яЁё]+', text))
    if not words:
        raise ValueError('Missing Cyrillic ending')
    start, end = words[-1].span()
    positions = [i for i,(a,b) in enumerate(encoded.offsets)
                 if a < end and b > start and labels[i] != -100]
    if not positions or min(positions) < 1:
        raise ValueError('Ending is outside supervised target')
    return positions


@torch.no_grad()
def ending_metrics(model, data, tokenizer, device, precision, limit):
    indices = np.linspace(0, len(data)-1, min(len(data),limit), dtype=np.int64)
    total_nll, targets, correct, words_correct = 0., 0, 0, 0
    for start in range(0,len(indices),16):
        batch = indices[start:start+16]
        ids, _ = data.batch(batch,device)
        with autocast(device,precision):
            logits = model(ids)
        for row,index in enumerate(batch):
            raw,labels = data.record(int(index))
            positions = ending_positions(raw,labels,tokenizer)
            scores = logits[row,torch.tensor(positions,device=device)-1].float()
            expected = ids[row,positions]
            losses = -scores.log_softmax(-1).gather(1,expected[:,None]).squeeze(1)
            if not torch.isfinite(losses).all():
                raise FloatingPointError('Nonfinite ending-word loss')
            hits = scores.argmax(-1).eq(expected)
            total_nll += float(losses.sum())
            targets += len(positions)
            correct += int(hits.sum())
            words_correct += int(hits.all())
    return dict(records=len(indices),targets=targets,nll=total_nll/targets,
                token_accuracy=correct/targets,teacher_forced_word_accuracy=words_correct/len(indices),
                correct_words=words_correct,
                definition='Final Cyrillic word subtokens only, each prediction sees the correct preceding context; not free-generation rhyme accuracy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    token_path = ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'
    token_hash, checkpoint_hash = sha256(token_path), sha256(args.checkpoint)
    manifest = read_json(args.data / 'manifest.json')
    if (manifest.get('serialization') != 'source_context_onlysecondline_newline_no_eos_loss'
            or token_hash not in [v for k,v in manifest['inputs_sha256'].items()
                                 if k.replace('\\', '/').endswith('tokenizer.json')]):
        raise ValueError('Expected frozen contextual data with selected tokenizer')
    token = Tokenizer.from_file(str(token_path))
    if any(token.token_to_id(k) != v for k,v in manifest['special_tokens'].items()):
        raise ValueError('Special token mismatch')
    datasets = {s: PreparedDataset(args.data, 'sft', s, 512) for s in ('train','validation')}
    with run_lock(ROOT/'runs/pretrain_series01', 'series'):
        if args.device == 'cuda':
            shared.AUDIT = ROOT/'audit/poetry16/gpu_admission.jsonl'
            shared.admit(torch, 3072, shared.Audit(phase='contextual16_evaluation',
                                                evaluator_sha256=sha256(Path(__file__))))
        state = load_checkpoint(args.checkpoint)
        if state['tokenizer_sha256'] != token_hash:
            raise ValueError('Checkpoint tokenizer mismatch')
        model = GPT(ModelConfig(**state['model_config']))
        model.load_state_dict(state['model'])
        model.to(args.device).eval()
        del state
        device = torch.device(args.device)
        precision = 'bf16' if device.type == 'cuda' else 'fp32'
        results = {s: evaluate(model, data, device, precision, batch_size=16,
                               max_batches=16 if s == 'train' else 0)
                   for s,data in datasets.items()}
        endings = {s: ending_metrics(model,data,token,device,precision,
                                     256 if s == 'train' else len(data))
                   for s,data in datasets.items()}
    if sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed during diagnostic')
    write_json(args.output, dict(schema='poetry16-contextual-nll-1', complete=True,
        training=False, test_used=False, checkpoint_sha256=checkpoint_hash,
        tokenizer_sha256=token_hash, manifest_sha256=sha256(args.data/'manifest.json'),
        evaluator_sha256=sha256(Path(__file__)), results=results, endings=endings,
        limitation='Fixed 256 train records versus full validation; different distributions, not a calibrated overfitting test. Second-line NLL is not rhyme or meaning quality.'))
    print({s:r['metrics']['overall'] for s,r in results.items()})


if __name__ == '__main__':
    main()

"""Generate from our own checkpoint; chat template matches frozen SFT."""
import argparse
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .serialization import encode_prompt
from .data import sha256
from .model import GPT, ModelConfig
from .runtime import autocast, load_checkpoint, resolve_device, seed_everything


def encode_continuation(tokenizer, text):
    # Pretraining has EOS between works; BOS appears only once per whole stream.
    return [tokenizer.token_to_id('<|eos|>')] + tokenizer.encode(text, add_special_tokens=False).ids


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=80, temperature=0.8, top_k=40, stop_ids=(), forbidden_ids=(), precision='fp32', *, top_p=1.0):
    if not prompt_ids or len(prompt_ids) >= model.config.context_length:
        raise ValueError('Prompt must leave room within context; silent truncation is forbidden')
    if max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError('Invalid generation parameters')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be finite and in (0, 1]')
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated, reason = [], 'max_new_tokens'
    for _ in range(max_new_tokens):
        if ids.shape[1] >= model.config.context_length:
            reason = 'context_limit'
            break
        with autocast(device, precision):
            logits = model(ids)[0, -1].float()
        logits[list(forbidden_ids)] = -torch.inf
        if temperature == 0:
            token = int(logits.argmax())
        else:
            logits /= temperature
            if top_k:
                threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < threshold] = -torch.inf
            if top_p < 1:
                logits = nucleus_logits(logits, top_p)
            token = int(torch.multinomial(logits.softmax(-1), 1))
        generated.append(token)
        if token in stop_ids:
            reason = 'stop_token'
            break
        ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
    return generated, reason


def nucleus_logits(logits, top_p):
    """Keep the crossing token; probabilities are normalized after top-k filtering."""
    if logits.ndim != 1 or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('Expected one logit vector and top_p in (0, 1]')
    if not torch.isfinite(logits).any() or torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError('Invalid sampling distribution')
    if top_p == 1:
        return logits
    ordered, indices = logits.sort(descending=True)
    probabilities = ordered.softmax(-1)
    keep = probabilities.cumsum(-1) - probabilities < top_p
    result = logits.clone()
    result[indices[~keep]] = -torch.inf
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path('artifacts/tokenizers_v3/selected/tokenizer.json'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', help='Raw pretrain continuation prefix')
    group.add_argument('--user', help='One user message rendered with the SFT template')
    group.add_argument('--messages', type=Path, help='JSON list ending in user; supports dialogue history')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = load_checkpoint(args.checkpoint)
    if sha256(args.tokenizer) != state['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from checkpoint')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if args.text is not None:
        prompt = encode_continuation(tokenizer, args.text)
    else:
        messages = json.loads(args.messages.read_text(encoding='utf-8-sig')) if args.messages else [{'role': 'user', 'content': args.user}]
        prompt = encode_prompt(tokenizer, messages)
    device, precision = resolve_device(args.device, args.precision)
    model = GPT(ModelConfig(**state['model_config'])).to(device)
    model.load_state_dict(state['model'])
    seed_everything(args.seed)
    stops = [tokenizer.token_to_id(t) for t in ('<|eos|>', '<|eot|>')]
    forbidden = [tokenizer.token_to_id(t) for t in ('<|pad|>', '<|bos|>', '<|user|>', '<|assistant|>')]
    ids, reason = generate(model, prompt, args.max_new_tokens, args.temperature, args.top_k, stops, forbidden, precision, top_p=args.top_p)
    print(json.dumps({'text': tokenizer.decode(ids, skip_special_tokens=True), 'stop_reason': reason,
                      'generated_tokens': len(ids)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

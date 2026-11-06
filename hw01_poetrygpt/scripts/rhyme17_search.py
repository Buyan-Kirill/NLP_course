"""Bounded search over model-generated prefixes and complete word suffixes.

Every accepted candidate is scored autoregressively from the unchanged prompt,
including all BPE pieces of the prescribed ending, punctuation and newline.
This is a more expensive inference system, not proof of learned free rhyme.
"""
import re

import torch
from poetrygpt.runtime import autocast
from scripts.rhyme_endings16 import last_word

WORD = re.compile(r'[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*')


def suffix_candidates(prompt_ids, prefix_texts, endings, tokenizer, *, body_limit=24,
                      min_prefix_words=3, max_prefix_words=12):
    """Branch only at complete Cyrillic word boundaries; never replace a word."""
    context = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    candidates = {}
    for draw, raw in enumerate(prefix_texts):
        first_line = raw.split('\n', 1)[0]
        words = list(WORD.finditer(first_line))
        if words and words[-1].end() == len(first_line) and '\n' not in raw:
            # A token-limited draw can finish halfway through a word. Do not
            # treat that final fragment as a confirmed branching boundary.
            words = words[:-1]
        for count in range(min_prefix_words, min(len(words), max_prefix_words)+1):
            prefix = first_line[:words[count-1].end()]
            if any(character in prefix for character in '\r\n'):
                raise ValueError('Prefix escaped its first line')
            for ending in endings:
                if WORD.fullmatch(ending) is None:
                    raise ValueError('Expected a whole Cyrillic ending')
                if (last_word(prefix) or '').replace('ё', 'е') == ending.lower().replace('ё', 'е'):
                    continue
                text = prefix + ' ' + ending + '.\n'
                full = tokenizer.encode(context+text, add_special_tokens=False).ids
                if full[:len(prompt_ids)] != prompt_ids:
                    raise ValueError('Candidate retokenized source context')
                ids = full[len(prompt_ids):]
                if len(ids) > body_limit or not ids:
                    continue
                if tokenizer.decode(full, skip_special_tokens=False) != context+text:
                    raise ValueError('Candidate failed text roundtrip')
                key = tuple(ids)
                if key not in candidates:
                    candidates[key] = dict(text=text, body_ids=ids, ending=ending,
                                            prefix_words=count, prefix_draw=draw)
    return list(candidates.values())


@torch.no_grad()
def score_candidates(model, prompt_ids, candidates, *, device, precision, batch_size=16):
    """Mean untempered log probability of the whole body including its stop."""
    if not candidates:
        return []
    rows = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start+batch_size]
        lengths = [len(prompt_ids)+len(c['body_ids']) for c in chunk]
        if max(lengths) > model.config.context_length:
            raise ValueError('Search would truncate context')
        ids = torch.zeros((len(chunk), max(lengths)), dtype=torch.long, device=device)
        for i, c in enumerate(chunk):
            ids[i, :lengths[i]] = ids.new_tensor(prompt_ids+c['body_ids'])
        with autocast(device, precision):
            logits = model(ids)
        for i, c in enumerate(chunk):
            scores = logits[i, len(prompt_ids)-1:lengths[i]-1].float()
            target = ids[i, len(prompt_ids):lengths[i]]
            logp = scores.log_softmax(-1).gather(1, target[:, None]).squeeze(1)
            if not torch.isfinite(logp).all():
                raise FloatingPointError('Nonfinite candidate log probability')
            rows.append(dict(c, logp_sum=float(logp.sum()), targets=len(target),
                             mean_logp=float(logp.mean())))
    return rows


def select_candidate(rows):
    if not rows:
        return None
    # A deterministic tie-break makes selection reproducible without an oracle.
    return max(rows, key=lambda r: (r['mean_logp'], r['logp_sum'], r['text']))

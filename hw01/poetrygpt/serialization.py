"""Explicit causal-LM input contract; no model or GPU dependency.

Labels are UNSHIFTED: the trainer must compare logits[:, :-1] with labels[:, 1:].
Pretrain permits attention across EOS document boundaries. SFT is not packed.
"""

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|eot|>"]
IGNORE = -100


def special_ids(tokenizer):
    result = {name: tokenizer.token_to_id(name) for name in SPECIAL_TOKENS}
    if any(value is None for value in result.values()):
        raise ValueError("Tokenizer is missing a required special token")
    return result


def encode_content(tokenizer, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Content must be a nonempty string")
    if any(marker in text for marker in SPECIAL_TOKENS):
        raise ValueError("Literal special-token marker in content")
    return tokenizer.encode(text, add_special_tokens=False).ids


def encode_sft(tokenizer, messages, max_length=512):
    special = special_ids(tokenizer)
    ids = [special["<|bos|>"]]
    labels = [IGNORE]
    if not messages or len(messages) % 2:
        raise ValueError("Expected complete alternating user/assistant exchanges")
    for position, message in enumerate(messages):
        role = "user" if position % 2 == 0 else "assistant"
        if message["role"] != role:
            raise ValueError("Unexpected message role")
        content = encode_content(tokenizer, message["content"])
        ids.append(special[f"<|{role}|>"])
        labels.append(IGNORE)
        ids.extend(content + [special["<|eot|>"]])
        labels.extend(content + [special["<|eot|>"]] if role == "assistant" else [IGNORE] * (len(content) + 1))
    ids.append(special["<|eos|>"])
    labels.append(special["<|eos|>"])
    if len(ids) > max_length:
        raise ValueError(f"SFT length {len(ids)} exceeds {max_length}; truncation is forbidden")
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def encode_prompt(tokenizer, messages):
    """Inference prefix ending at assistant role; messages end in a user turn."""
    special = special_ids(tokenizer)
    if not messages or len(messages) % 2 != 1:
        raise ValueError("Inference messages must end in user")
    ids = [special["<|bos|>"]]
    for position, message in enumerate(messages):
        role = "user" if position % 2 == 0 else "assistant"
        if message["role"] != role:
            raise ValueError("Unexpected role")
        ids.extend([special[f"<|{role}|>"]] + encode_content(tokenizer, message["content"]) + [special["<|eot|>"]])
    return ids + [special["<|assistant|>"]]


def pad_batch(records, pad_id, length=None):
    if not records:
        raise ValueError("Empty batch")
    longest = max(len(record["input_ids"]) for record in records)
    length = longest if length is None else length
    if length < longest:
        raise ValueError("Padding must not truncate")
    batch = {key: [] for key in ("input_ids", "labels", "attention_mask")}
    for record in records:
        n = len(record["input_ids"])
        if len(record["labels"]) != n or len(record["attention_mask"]) != n:
            raise ValueError("Unaligned sequence fields")
        batch["input_ids"].append(record["input_ids"] + [pad_id] * (length - n))
        batch["labels"].append(record["labels"] + [IGNORE] * (length - n))
        batch["attention_mask"].append(record["attention_mask"] + [0] * (length - n))
    return batch


def pretrain_stream(tokenizer, records):
    """One BOS per split, EOS after every work; metadata stays in sidecar offsets."""
    special = special_ids(tokenizer)
    ids = [special["<|bos|>"]]
    offsets = []
    for record in records:
        start = len(ids)
        ids.extend(encode_content(tokenizer, record["text"]))
        ids.append(special["<|eos|>"])
        offsets.append({"work_id": record["work_id"], "start": start, "end": len(ids)})
    return ids, offsets


def pretrain_windows(ids, context_length=512):
    """Overlap one token: every token after initial BOS is a target exactly once.

    Last short window is retained. First label is masked because causal shift
    discards it; callers pad windows using pad_batch rather than drop_last.
    """
    if context_length < 2:
        raise ValueError("Context must allow at least one target")
    for start in range(0, len(ids) - 1, context_length - 1):
        chunk = list(ids[start:start + context_length])
        yield {"input_ids": chunk, "labels": [IGNORE] + chunk[1:], "attention_mask": [1] * len(chunk)}

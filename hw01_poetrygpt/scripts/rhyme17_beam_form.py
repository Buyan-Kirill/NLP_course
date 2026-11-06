"""Version2: suffix-aware search with short repetition and syllable-band filters.

Prefix alternatives are explored while the prescribed ending is already known.
The model scores every complete BPE path through ending, punctuation and newline.
This heuristic search is not exact global optimization or a semantic judge.
"""
import torch
from poetrygpt.runtime import autocast
from scripts.rhyme17_search import score_candidates,select_candidate,WORD
from scripts.rhyme_endings16 import last_word,VOWELS


def completions(prompt,prefixes,ending,tokenizer,body_limit,min_prefix_words,word_ngram=2,syllable_range=None):
    context=tokenizer.decode(prompt,skip_special_tokens=False)
    rows={}
    for state in prefixes:
        prefix=tokenizer.decode(state['ids'],skip_special_tokens=False)
        if '\n' in prefix or '\ufffd' in prefix or len(WORD.findall(prefix))<min_prefix_words:
            continue
        prefix=prefix.rstrip()
        if not prefix or not prefix[-1].isalpha():
            continue
        if (last_word(prefix) or '').replace('ё','е')==ending.lower().replace('ё','е'):
            text=prefix+'.\n'
        else:
            text=prefix+' '+ending+'.\n'
        if repeated_words(text,word_ngram):
            continue
        syllables=sum(c.lower() in VOWELS for c in text)
        if syllable_range is not None and not syllable_range[0]<=syllables<=syllable_range[1]:
            continue
        ids=tokenizer.encode(context+text,add_special_tokens=False).ids
        if ids[:len(prompt)]!=prompt:raise ValueError('Beam retokenized prompt')
        body=ids[len(prompt):]
        if not 0<len(body)<=body_limit:continue
        rows[tuple(body)]=dict(text=text,body_ids=body,ending=ending,
                               prefix_words=len(WORD.findall(prefix)))
    return list(rows.values())


def repeated_words(text,n=2):
    words=WORD.findall(text.lower().replace('ё','е'))
    grams=[tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))]
    return len(set(grams))<len(grams)


@torch.no_grad()
def search(model,prompt,ending,tokenizer,*,device,precision,forbidden_ids,
           width=8,branch=8,max_prefix_tokens=16,body_limit=24,min_prefix_words=2,
           word_ngram=2,syllable_range=None):
    if (WORD.fullmatch(ending) is None or not prompt
            or any(type(n) is not int or n<1 for n in
                   [width,branch,max_prefix_tokens,body_limit,min_prefix_words,word_ngram])):
        raise ValueError('Invalid whole-word ending or bounded search configuration')
    if syllable_range is not None and (len(syllable_range)!=2
            or any(type(n) is not int or n<1 for n in syllable_range)
            or syllable_range[0]>syllable_range[1]):
        raise ValueError('Invalid syllable band')
    if len(prompt)+body_limit>model.config.context_length:
        raise ValueError('Beam would truncate context')
    if model.training:raise ValueError('Beam requires eval mode')
    blocked=set(forbidden_ids)
    blocked.update(i for i in range(tokenizer.get_vocab_size())
                   if '\n' in tokenizer.decode([i],skip_special_tokens=False)
                   or '\r' in tokenizer.decode([i],skip_special_tokens=False))
    beams=[dict(ids=[],logp=0.)];finished={};trace=[];scored_positions=0
    cache={}
    for depth in range(max_prefix_tokens):
        full=[prompt+b['ids'] for b in beams]
        lengths=[len(ids) for ids in full]
        batch=torch.zeros((len(full),max(lengths)),dtype=torch.long,device=device)
        for row,ids in enumerate(full):batch[row,:len(ids)]=batch.new_tensor(ids)
        with autocast(device,precision):logits=model(batch)
        expanded=[]
        for row,state in enumerate(beams):
            scores=logits[row,lengths[row]-1].float().log_softmax(-1)
            scores[list(blocked)]=-torch.inf
            values,indices=scores.topk(min(branch,int(torch.isfinite(scores).sum())))
            for value,index in zip(values.tolist(),indices.tolist()):
                ids=state['ids']+[index]
                text=tokenizer.decode(ids,skip_special_tokens=False)
                if repeated_words(text,word_ngram):continue
                if syllable_range is not None and sum(c.lower() in VOWELS for c in text)>syllable_range[1]:continue
                expanded.append(dict(ids=ids,logp=state['logp']+value))
        if not expanded:break
        candidates=completions(prompt,expanded,ending,tokenizer,body_limit,min_prefix_words,word_ngram,syllable_range)
        fresh=[r for r in candidates if tuple(r['body_ids']) not in cache]
        scored=score_candidates(model,prompt,fresh,device=device,precision=precision)
        for r in scored:cache[tuple(r['body_ids'])]=r
        scored_positions+=sum(r['targets'] for r in scored)
        for r in candidates:finished[tuple(r['body_ids'])]=cache[tuple(r['body_ids'])]
        # Ending-aware lookahead guides prefix exploration. A prefix that has no
        # eligible complete suffix yet retains its ordinary average logp score.
        ranked=[]
        for state in expanded:
            options=completions(prompt,[state],ending,tokenizer,body_limit,min_prefix_words,word_ngram,syllable_range)
            best=select_candidate([cache[tuple(r['body_ids'])] for r in options])
            rank=best['mean_logp'] if best else state['logp']/len(state['ids'])
            ranked.append((rank,state['logp'],tuple(state['ids']),state))
        ranked.sort(key=lambda x:x[:3],reverse=True)
        # Preserve one ordinary high-probability prefix: an early suffix
        # lookahead can otherwise prune a good sentence before its verb/object
        # has been generated. Remaining slots explore ending-aware branches.
        ordinary=max(expanded,key=lambda r:(r['logp'],tuple(r['ids'])))
        beams=[ordinary]
        for _,_,_,state in ranked:
            if state['ids']!=ordinary['ids'] and len(beams)<width:
                beams.append(state)
        trace.append(dict(depth=depth+1,expanded=len(expanded),kept=len(beams),
                          complete_candidates=len(finished)))
    rows=list(finished.values())
    return dict(selected=select_candidate(rows),candidates=rows,trace=trace,
                scored_body_positions=scored_positions,settings=dict(width=width,branch=branch,
                max_prefix_tokens=max_prefix_tokens,body_limit=body_limit,
                min_prefix_words=min_prefix_words,no_repeated_word_ngram=word_ngram,syllable_range=syllable_range))

"""Bounded native-context beam over a set of complete rhyme words.

All endings participate while prefixes are built. No chosen W or plan marker
enters the prompt. This heuristic is neither an exact search nor a meaning judge.
"""
import torch

from poetrygpt.runtime import autocast
from scripts.rhyme17_beam_form import completions, repeated_words
from scripts.rhyme17_search import WORD, score_candidates, select_candidate
from scripts.reselect_rhyme17_repetitions import repeated_content_words
from scripts.rhyme_endings16 import last_word, VOWELS


@torch.no_grad()
def search(model, prompt, endings, tokenizer, *, device, precision,
           forbidden_ids, syllable_range, width=8, branch=8,
           max_prefix_tokens=16, body_limit=24, min_prefix_words=2):
    if (model.training or not prompt or not endings or len(endings)>32
            or len(set(endings))!=len(endings)
            or any(WORD.fullmatch(word) is None for word in endings)
            or any(type(n) is not int or n<1 for n in
                   (width,branch,max_prefix_tokens,body_limit,min_prefix_words))
            or max_prefix_tokens>body_limit
            or len(prompt)+body_limit>model.config.context_length
            or len(syllable_range)!=2 or not 1<=syllable_range[0]<=syllable_range[1]):
        raise ValueError('Invalid bounded native joint search')
    if any(type(i) is not int or not 0<=i<tokenizer.get_vocab_size() for i in prompt):
        raise ValueError('Invalid native prompt IDs')
    context=tokenizer.decode(prompt,skip_special_tokens=False)
    if tokenizer.encode(context,add_special_tokens=False).ids!=prompt:
        raise ValueError('Native prompt is not a canonical BPE roundtrip')
    blocked=set(forbidden_ids)
    blocked.update(i for i in range(tokenizer.get_vocab_size())
        if '\n' in tokenizer.decode([i],skip_special_tokens=False)
        or '\r' in tokenizer.decode([i],skip_special_tokens=False))
    beams=[dict(ids=[],logp=0.)]
    finished, trace = {}, []
    scored_positions=0
    for depth in range(max_prefix_tokens):
        full=[prompt+b['ids'] for b in beams]
        batch=torch.zeros((len(full),max(map(len,full))),dtype=torch.long,device=device)
        for index,ids in enumerate(full):
            batch[index,:len(ids)]=batch.new_tensor(ids)
        with autocast(device,precision):
            logits=model(batch)
        expanded=[]
        for index,state in enumerate(beams):
            scores=logits[index,len(full[index])-1].float().log_softmax(-1)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite native beam probabilities')
            scores[list(blocked)]=-torch.inf
            count=min(branch,int(torch.isfinite(scores).sum()))
            values,indices=scores.topk(count)
            for value,token in zip(values.tolist(),indices.tolist()):
                ids=state['ids']+[token]
                text=tokenizer.decode(ids,skip_special_tokens=False)
                if (repeated_words(text,2)
                        or sum(c.lower() in VOWELS for c in text)>syllable_range[1]):
                    continue
                expanded.append(dict(ids=ids,logp=state['logp']+value))
        if not expanded:
            break
        options_by_prefix=[]
        new={}
        rejected=0
        for state in expanded:
            options={}
            for ending in endings:
                for row in completions(prompt,[state],ending,tokenizer,body_limit,
                        min_prefix_words,2,syllable_range):
                    if repeated_content_words(row['text']):
                        rejected+=1
                        continue
                    if (tokenizer.decode(prompt+row['body_ids'],skip_special_tokens=False)!=context+row['text']
                            or tokenizer.encode(context+row['text'],add_special_tokens=False).ids!=prompt+row['body_ids']
                            or (last_word(row['text']) or '').casefold().replace('ё','е')!=ending.casefold().replace('ё','е')):
                        raise ValueError('Complete native path/whole-ending roundtrip failed')
                    key=tuple(row['body_ids'])
                    options[key]=row
                    if key not in finished:
                        new[key]=row
            options_by_prefix.append(options)
        scored=score_candidates(model,prompt,list(new.values()),device=device,precision=precision)
        scored_positions+=sum(row['targets'] for row in scored)
        finished.update((tuple(row['body_ids']),row) for row in scored)
        ranked=[]
        for state,options in zip(expanded,options_by_prefix):
            best=select_candidate([finished[key] for key in options])
            rank=best['mean_logp'] if best else state['logp']/len(state['ids'])
            ranked.append((rank,state['logp'],tuple(state['ids']),state))
        ranked.sort(key=lambda r:r[:3],reverse=True)
        ordinary=max(expanded,key=lambda r:(r['logp'],tuple(r['ids'])))
        beams=[ordinary]
        for _,_,_,state in ranked:
            if state['ids']!=ordinary['ids'] and len(beams)<width:
                beams.append(state)
        trace.append(dict(depth=depth+1,expanded=len(expanded),kept=len(beams),
            new_candidates=len(scored),total_candidates=len(finished),
            lexical_repeat_rejections=rejected))
    candidates=list(finished.values())
    return dict(selected=select_candidate(candidates),candidates=candidates,trace=trace,
        scored_body_positions=scored_positions,
        settings=dict(width=width,branch=branch,max_prefix_tokens=max_prefix_tokens,
            body_limit=body_limit,min_prefix_words=min_prefix_words,
            syllable_range=syllable_range,pool_size=len(endings),plan_marker_used=False,
            maximum_candidate_constructions=max_prefix_tokens*width*branch*len(endings),
            selection='Native full-body mean LP with unchanged lexical-repeat filter'))

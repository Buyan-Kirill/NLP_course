"""Word-frequency and writing-system statistics for the training corpus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from reproduction.download import ROOT

WORDS = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", re.UNICODE)
CYRILLIC = re.compile(r'[А-Яа-яЁёѢѣІіѲѳѴѵ]')
LATIN = re.compile(r'[A-Za-z]')


def analyze(path):
    words, lengths = Counter(), []
    digest = hashlib.sha256()
    characters = cyrillic = latin = records = 0
    with Path(path).open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            text = row['text']
            counts = Counter(word.lower() for word in WORDS.findall(text))
            words.update(counts)
            lengths.append(sum(counts.values()))
            characters += len(text)
            cyrillic += len(CYRILLIC.findall(text))
            latin += len(LATIN.findall(text))
            records += 1
    ordered = sorted(lengths)
    def percentile(fraction):
        return ordered[round((len(ordered)-1)*fraction)] if ordered else 0
    total = sum(words.values())
    head = words.most_common(20)
    return dict(input_sha256=digest.hexdigest(), records=records, characters=characters,
        words=total, word_types=len(words), top20=[dict(word=w,count=n) for w,n in head],
        top20_token_share=sum(n for _,n in head)/max(1,total),
        word_frequency_types={
            '1':sum(n==1 for n in words.values()),
            '2-5':sum(2<=n<=5 for n in words.values()),
            '6-20':sum(6<=n<=20 for n in words.values()),
            '>20':sum(n>20 for n in words.values())},
        document_words=dict(min=min(lengths,default=0),p50=percentile(.5),
                            p95=percentile(.95),max=max(lengths,default=0)),
        script_letters=dict(cyrillic=cyrillic,latin=latin,
            latin_share_of_cyrillic_and_latin=latin/max(1,cyrillic+latin)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prose',type=Path,default=ROOT/'data/prose_v7/expanded_train.jsonl')
    parser.add_argument('--poetry',type=Path,default=ROOT/'data/poetry_v7/train.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/corpus_statistics.json')
    args=parser.parse_args()
    report=dict(definition='Lowercase Unicode words; internal hyphen/apostrophe retained; no lemmatization',
        scope='Final train only; script counts are diagnostics, not language identification',domains={})
    for domain,path in [('prose',args.prose),('poetry',args.poetry)]:
        report['domains'][domain]=analyze(path)
        print(domain,report['domains'][domain]['records'],'records',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8',newline='\n')
    print(args.output)


if __name__=='__main__':
    main()

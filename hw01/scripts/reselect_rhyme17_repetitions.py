"""Re-select existing complete beam paths after a fixed lexical repeat filter.

No generation/training and no semantic score. This is open-dev tuning motivated
by repeated content words surviving the existing bigram constraint.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rhyme17_search import WORD, select_candidate

FUNCTION_WORDS = frozenset('как что где кто это этой этот этих всё все всех для при или ибо мой моя мои мне мной мною твой твоя твои твой твоей тебя тебе тобой его ему ею неё нее ней она оно они она нас нам вами вам вас уже ещё еще лишь вот там тут так был была были было нет над под без про чтоб чтобы если когда ведь пусть даже разве'.split())


def repeated_content_words(text):
    words = [w.casefold().replace('ё', 'е') for w in WORD.findall(text)]
    counts = Counter(w for w in words if len(w) >= 3 and w not in FUNCTION_WORDS)
    return sorted(w for w, count in counts.items() if count > 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding='utf-8'))
    if source.get('complete') is not True:
        raise ValueError('Complete original search required')
    rows = []
    for case in source['searches']:
        # Both frozen form output and contextual-word-ranked output retain
        # complete candidate paths inside each ending alternative.
        alternatives = case['alternatives']
        case_id = case.get('case_id', case.get('case', {}).get('id'))
        first_line = next(r['first_line'] for r in source['samples'] if r['case_id'] == case_id)
        candidates = [r for alternative in alternatives for r in alternative['beam']['candidates']]
        accepted = [r for r in candidates if not repeated_content_words(r['text'])]
        selected = select_candidate(accepted)
        rows.append(dict(case_id=case_id, first_line=first_line,
            mode=case.get('mode', 'dictionary'), kind='lexical_repeat_reselection',
            oracle=case.get('mode') == 'supplied', selected=selected,
            complete_candidates=len(candidates), accepted_candidates=len(accepted),
            rejected_repeated_content=len(candidates)-len(accepted)))
    report = dict(schema='rhyme17-existing-path-lexical-reselection-1', complete=True,
        training=False, generation=False, source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rule='Reject exact normalized repeated words length>=3 outside explicit function-word list; select unchanged mean body LP.',
        function_words=sorted(FUNCTION_WORDS), samples=rows,
        limitation='Not a grammar or meaning filter; repeated inflected forms and different nonsense can survive. Open-dev exploratory tuning.')
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output.exists():
        raise FileExistsError('Preserve existing selection')
    args.output.write_text(content, encoding='utf-8')
    for row in rows:
        print(row['mode'], row['case_id'], (row['selected'] or {}).get('text'))


if __name__ == '__main__':
    main()

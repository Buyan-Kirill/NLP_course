"""Interactive rhyme pools with an explicit reading of ambiguous input stress.

The frozen evaluator/search stay unchanged. The new policy conditions its
exact-rhyme proof on one input reading, retaining all original alternatives.
Candidate words must still rhyme under every dictionary reading of that word.
"""
import argparse
from copy import deepcopy
from functools import lru_cache
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_rhyme17_runtime_pool as legacy
from scripts.rhyme_endings16 import VOWELS, compare_endings, lexical_stresses, last_word
from poetrygpt.data import read_json, sha256
from poetrygpt.runtime import write_json


def explicit_stress(line):
    """Read an optional combining acute on the last lexical word (1-based)."""
    words = re.findall('[а-яё\u0301\u0341]+', line.casefold())
    if not words:
        return None
    word = words[-1]
    indices = [i for i, c in enumerate(word) if c in '\u0301\u0341']
    if not indices:
        return None
    if len(indices) != 1 or indices[0] == 0 or word[indices[0] - 1] not in VOWELS:
        raise ValueError('Put one stress mark immediately after a vowel in the final word')
    return sum(c in VOWELS for c in word[:indices[0]])


def make_comparator(tool, inference_mode):
    @lru_cache(maxsize=64)
    def reading(left):
        word = last_word(left)
        with inference_mode():
            positions, source = lexical_stresses(tool.accentuator, word)
            explicit = explicit_stress(left)
            if explicit is not None:
                # Explicit user pronunciation takes priority over dictionary variants.
                return explicit, 'user_accent', positions
            if len(positions) > 1:
                if source == 'ambiguous_dictionary':
                    tokens = [t for parsing in tool.udpipe.parse_text(left) for t in parsing]
                    token = next((t for t in reversed(tokens) if t.form.casefold() == word), None)
                    if token is not None:
                        chosen = tool.accentuator.predict_ambiguous_accent(word, [token.upos, *token.tags])
                        if chosen in positions:
                            return chosen, 'rpst_morphology_dictionary', positions
                predicted = tool.accentuator.predict_stress(word)
                if predicted in positions:
                    return predicted, 'rpst_stress_model_among_dictionary_variants', positions
            return None, 'all_dictionary_readings', positions

    def compare(left, right):
        with inference_mode():
            result = compare_endings(tool, left, right)
        if result['status'] != 'ok':
            return result
        position, method, alternatives = reading(left)
        if position is None:
            return result
        original = deepcopy(result)
        combinations = [c for c in result['combinations'] if c['stresses'][0] == position]
        if not combinations:
            # Explicit accents outside the lexicon need a fresh phonetic proof.
            from russian_scansion.phonetic import rhymed2, rhymed_fuzzy2
            for candidate_stress in result['stress_evidence'][1][0]:
                args = (tool.accentuator, result['words'][0], position, [], '', '',
                        result['words'][1], candidate_stress, [], '', '')
                combinations.append(dict(stresses=[position, candidate_stress],
                    exact=bool(rhymed2(*args)), fuzzy=bool(rhymed_fuzzy2(*args))))
        result.update(combinations=combinations,
            stress_evidence=[[[position], method], result['stress_evidence'][1]],
            exact_any=any(c['exact'] for c in combinations),
            exact_all=all(c['exact'] for c in combinations),
            fuzzy_any=any(c['fuzzy'] for c in combinations),
            input_reading=dict(position=position, method=method, alternatives=alternatives,
                context_disambiguated=method == 'rpst_morphology_dictionary', automatic=method != 'user_accent'),
            unresolved_evidence=original)
        return result
    return compare


def load_phonetics():
    from scripts.probe_rpst08 import load_verified_tool
    tool, torch, version = load_verified_tool(legacy.ARTIFACTS)
    return make_comparator(tool, torch.inference_mode), dict(
        rpst_revision=version['rpst_revision'], rpst_version=version['rpst_version'], device='cpu',
        input_stress_policy='user acute; morphology for homographs; stress model for variants',
        limitation='Automatic stress and morphology can be wrong; user acute overrides them')


def prepare(input_path, output):
    output = legacy.local(output)
    staging = output.with_suffix('.building22.json')
    if output.exists() or staging.exists():
        raise FileExistsError('Preserve existing inference artifacts')
    own_path = Path(__file__).resolve()
    own_sha = sha256(own_path)
    legacy.prepare(input_path, staging, phonetics_factory=load_phonetics)
    report = read_json(staging)
    if report['split'] != 'inference' or report['test_used']:
        raise ValueError('Interactive builder does not open evaluation sets')
    report['policy'].update(phonetic='RPST exact_all conditional on the recorded input stress',
        input_stress_resolution='explicit accent, RPST morphology, or stress prediction; alternatives retained',
        runtime_version=22)
    report['sources_sha256'][own_path.relative_to(ROOT).as_posix()] = own_sha
    if sha256(own_path) != own_sha:
        raise ValueError('Builder changed while running')
    write_json(staging, report)
    os.link(staging, output)  # Publish complete output without overwriting a prior run.
    staging.unlink()
    return dict(output=legacy.relative(output), sha256=sha256(output), status_counts=report['status_counts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.input, args.output))

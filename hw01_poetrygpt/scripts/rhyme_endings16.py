"""Word-final rhyme independent of RPST's choice of metrical line stresses.

This complements whole-line scansion. Ambiguous lexical stresses remain explicit;
no probability, semantic judgement or historical pronunciation guarantee is made.
"""
from itertools import product
import re

VOWELS = 'аеёиоуыэюя'


def last_word(line):
    words = re.findall('[а-яё]+', line.lower().replace('\u0301', '').replace('\u0341', ''))
    return words[-1] if words else None


def lexical_stresses(accentuator, word):
    if sum(c in VOWELS for c in word) == 1:
        return [1], 'one_vowel'
    ambiguous = accentuator.ambiguous_accents or {}
    if word in ambiguous:
        variants = []
        for spelling in ambiguous[word]:
            count = 0
            for c in spelling:
                count += c.lower() in VOWELS
                if c in VOWELS.upper():
                    variants.append(count)
                    break
        if variants:
            return sorted(set(variants)), 'ambiguous_dictionary'
    if word in accentuator.ambiguous_accents2:
        return sorted(set(accentuator.ambiguous_accents2[word])), 'variant_dictionary'
    source = 'dictionary' if word in accentuator.word_accents_dict else 'predicted_or_normalized'
    stress = accentuator.get_accent(word, [])
    return ([stress] if type(stress) is int and stress > 0 else []), source


def compare_endings(tool, left, right):
    from russian_scansion.phonetic import rhymed2, rhymed_fuzzy2
    words = [last_word(left), last_word(right)]
    if None in words:
        return dict(status='unknown', words=words, reason='missing_cyrillic_ending')
    if words[0].replace('ё', 'е') == words[1].replace('ё', 'е'):
        return dict(status='identical_word', words=words, exact_any=False, exact_all=False)
    try:
        stresses = [lexical_stresses(tool.accentuator, word) for word in words]
        if any(not positions for positions, source in stresses):
            return dict(status='unknown', words=words, stresses=stresses, reason='missing_stress')
        combinations = []
        for a, b in product(stresses[0][0], stresses[1][0]):
            args = (tool.accentuator, words[0], a, [], '', '', words[1], b, [], '', '')
            combinations.append(dict(stresses=[a, b], exact=bool(rhymed2(*args)),
                                     fuzzy=bool(rhymed_fuzzy2(*args))))
        return dict(status='ok', words=words, stress_evidence=stresses, combinations=combinations,
            exact_any=any(c['exact'] for c in combinations), exact_all=all(c['exact'] for c in combinations),
            fuzzy_any=any(c['fuzzy'] for c in combinations))
    except Exception as error:
        return dict(status='error', words=words, diagnostic=f'{type(error).__name__}: {error}')

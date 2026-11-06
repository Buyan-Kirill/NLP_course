"""Regression: resolve input stress without weakening candidate rhyme proofs."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_rhyme22_runtime_pool as pool


def evidence(candidate_ambiguous=False):
    combinations = [dict(stresses=[1, 2], exact=False, fuzzy=False),
                    dict(stresses=[2, 2], exact=True, fuzzy=True)]
    if candidate_ambiguous:
        combinations.append(dict(stresses=[2, 1], exact=False, fuzzy=False))
    return dict(status='ok', words=['саду', 'найду'],
        stress_evidence=[[[1, 2], 'ambiguous_dictionary'], [[2, 1] if candidate_ambiguous else [2], 'dictionary']],
        combinations=combinations, exact_any=True, exact_all=False, fuzzy_any=True)


class StressTests(unittest.TestCase):
    def comparator(self):
        token = SimpleNamespace(form='саду', upos='NOUN', tags=['Case=Loc'])
        accentuator = SimpleNamespace(predict_ambiguous_accent=lambda *args: 2,
                                     predict_stress=lambda word: 1)
        tool = SimpleNamespace(accentuator=accentuator,
            udpipe=SimpleNamespace(parse_text=lambda text: [[token]]))
        return pool.make_comparator(tool, nullcontext)

    def test_context_resolves_input_and_original_evidence_survives(self):
        with patch.object(pool, 'lexical_stresses', return_value=([1, 2], 'ambiguous_dictionary')), \
             patch.object(pool, 'compare_endings', side_effect=lambda *args: evidence()):
            result = self.comparator()('Один стою в саду', 'найду')
        self.assertTrue(result['exact_all'])
        self.assertFalse(result['unresolved_evidence']['exact_all'])
        self.assertEqual(result['input_reading']['position'], 2)
        self.assertTrue(result['input_reading']['context_disambiguated'])

    def test_does_not_relax_candidate_ambiguity(self):
        with patch.object(pool, 'lexical_stresses', return_value=([1, 2], 'ambiguous_dictionary')), \
             patch.object(pool, 'compare_endings', side_effect=lambda *args: evidence(True)):
            self.assertFalse(self.comparator()('Один стою в саду', 'найду')['exact_all'])

    def test_user_accent_and_identical_words(self):
        with patch.object(pool, 'lexical_stresses', return_value=([1, 2], 'ambiguous_dictionary')), \
             patch.object(pool, 'compare_endings', side_effect=lambda *args: evidence()):
            result = self.comparator()('Один стою в са́ду', 'найду')
        self.assertFalse(result['exact_all'])
        self.assertEqual(result['input_reading']['method'], 'user_accent')
        with patch.object(pool, 'compare_endings', return_value=dict(status='identical_word')):
            self.assertEqual(self.comparator()('саду', 'саду')['status'], 'identical_word')

    def test_acute_validation(self):
        self.assertEqual(pool.explicit_stress('Стою в саду́.'), 2)
        self.assertIsNone(pool.explicit_stress('Стою в саду.'))
        for line in ('Стою в са́ду́.', 'Стою в сад́у.'):
            with self.assertRaises(ValueError):
                pool.explicit_stress(line)


if __name__ == '__main__':
    unittest.main()

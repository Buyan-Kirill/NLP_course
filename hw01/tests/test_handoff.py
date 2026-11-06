"""Offline delivery checks: archive safety, source spans and public workflow."""
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid
import unittest
from unittest.mock import patch
import zipfile

from reproduction.datasets import select_text
from reproduction.download import extract_checked, fetch, https_context, ROOT
from reproduction.run import STAGES, recipe


@contextmanager
def fixture():
    root = ROOT / 'tests' / ('_delivery_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        assert root.resolve().parent == (ROOT / 'tests').resolve()
        shutil.rmtree(root)


class DeliveryTests(unittest.TestCase):
    def test_report_links_work_without_sibling_files_and_keep_local_anchors(self):
        from reproduction.build_report import portable_links, REPOSITORY
        source = '<a href="../../README.md">Code</a><a href="#metrics">Metrics</a>'
        result = portable_links(source)
        self.assertIn('href="'+REPOSITORY+'README.md"', result)
        self.assertIn('href="#metrics"', result)
        self.assertNotIn('href="../', result)

    def test_corpus_word_statistics_use_word_types_and_record_lengths(self):
        from reproduction.analyze_corpus import analyze
        with fixture() as folder:
            path = folder / 'sample.jsonl'
            path.write_text(json.dumps(dict(text='Лес, лес. Ёж — жук-плавунец! tea.'), ensure_ascii=False)+'\n', 'utf8')
            stats = analyze(path)
        self.assertEqual(stats['words'], 5)
        self.assertEqual(stats['word_types'], 4)
        self.assertEqual(stats['word_frequency_types']['1'], 3)
        self.assertEqual(stats['document_words']['p50'], 5)
        self.assertEqual(stats['script_letters']['latin'], 3)
        self.assertEqual(stats['top20'][0], dict(word='лес', count=2))

    def test_npz_bytes_match_the_frozen_layout_across_platforms(self):
        import numpy as np
        from reproduction.array_archive import write_npz
        arrays = dict(input_ids=np.array([[1,7,2,0],[1,8,9,2]], dtype=np.uint16),
                      labels=np.array([[-100,7,2,-100],[-100,8,9,2]], dtype=np.int32),
                      attention_mask=np.array([[1,1,1,0],[1,1,1,1]], dtype=np.uint8))
        with fixture() as root:
            path = root / 'records.npz'
            write_npz(path, arrays)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             '8c48bb5ae8d9d4638e48d344f0537c5de1b6330fcbb5baea6dde680daa3ed776')
            with np.load(path, allow_pickle=False) as restored:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(restored[key], value)

    def test_download_keeps_certificate_and_hostname_verification(self):
        import ssl
        context = https_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_source_selection_preserves_unicode_and_newlines(self):
        source = 'Предисловие.\nОдин стою в саду.\n\nИ тихо вечера я жду.\nСноска.'
        first = source.index('Один'); last = source.index('\nСноска')
        selection = dict(pieces=[[first, last]])
        self.assertEqual(select_text(source, selection), 'Один стою в саду.\n\nИ тихо вечера я жду.')
        with self.assertRaises(ValueError):
            select_text(source, dict(pieces=[[-1, 8]]))

    def test_batched_whole_work_encoding_matches_individual_encoding(self):
        import numpy as np
        from tokenizers import Tokenizer
        from reproduction import serialize
        token = Tokenizer.from_file(str(ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'))
        rows = [dict(work_id='w1', text='В саду светит месяц.\nУмолк давно весёлый смех.'),
                dict(work_id='w2', text='Дождь прошёл.\n\nИ воздух тих.')]
        metadata = {r['work_id']: dict(work_id=r['work_id'], family_id=r['work_id']) for r in rows}
        expected = [token.token_to_id('<|bos|>')]
        for row in rows:
            expected += token.encode(row['text'], add_special_tokens=False).ids + [token.token_to_id('<|eos|>')]
        with fixture() as folder, patch.dict(serialize.LINE_ENDINGS, {'poetry_train_offsets.json': '\r\n'}):
            stats, _ = serialize.prepare_pretrain(token, rows, metadata, 'poetry', 'train', out=folder)
            actual = np.load(folder / 'poetry_train.npy', allow_pickle=False)
            np.testing.assert_array_equal(actual, expected)
            offsets_path = folder / 'poetry_train_offsets.json'
            self.assertTrue(offsets_path.read_bytes().startswith(b'[\r\n'))
            offsets = json.loads(offsets_path.read_text('utf8'))
            for row, offset in zip(rows, offsets):
                self.assertEqual(token.decode(actual[offset['start']:offset['end']-1].tolist()), row['text'])
            self.assertEqual(stats['causal_targets'], len(expected)-1)

    def test_download_uses_only_checksum_verified_cache(self):
        with fixture() as folder:
            path = Path(folder) / 'source'
            path.write_bytes(b'original')
            digest = hashlib.sha256(b'original').hexdigest()
            with patch('reproduction.download.urlopen') as network:
                self.assertEqual(fetch('https://example.invalid', path, digest), path)
                network.assert_not_called()
                with self.assertRaises(ValueError):
                    fetch('https://example.invalid', path, '0' * 64)

    def test_archive_rejects_escape_before_writing(self):
        with fixture() as folder:
            root = Path(folder)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('good.txt', b'ok')
                stream.writestr('../escape', b'bad')
            files = [dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                     for name, data in [('good.txt', b'ok'), ('../escape', b'bad')]]
            with self.assertRaises(ValueError):
                extract_checked(archive, root / 'out', files)
            self.assertFalse((root / 'out/good.txt').exists())

    def test_selected_workflow_and_portable_wrapper(self):
        self.assertEqual(list(STAGES), ['pretrain', 'pretrain_finish', 'poetry',
                                        'poetry_small_batch', 'ending_plan'])
        for name in STAGES:
            cfg, command, _, _ = recipe(name)
            self.assertIn('reproduction.train_stage', command)
            self.assertTrue((ROOT / STAGES[name]['script']).is_file())
        cfg, _, _, _ = recipe('pretrain')
        self.assertEqual(cfg['optimizer'], 'muon')
        self.assertEqual(cfg['model']['vocab_size'], 8192)

    def test_selected_model_parameter_count(self):
        import torch
        from poetrygpt.model import GPT, ModelConfig
        cfg, _, _, _ = recipe('pretrain')
        with torch.device('meta'):
            model = GPT(ModelConfig(**cfg['model']))
        self.assertEqual(model.parameter_count(), 68440576)

    def test_html_footer_drift_is_accepted_but_body_change_is_not(self):
        from reproduction import datasets
        from reproduction.source_views import views
        raw = ('<meta charset="utf-8"><!--Собственно произведение-->'
               '<p>В саду под вечер свет горит.</p></noindex>counter=1').encode('utf8')
        body = views(raw, 'Lib.ru_Classic', requested='lib_legacy')['lib_legacy']
        expected = hashlib.sha256(raw).hexdigest()
        key = expected[:24]
        spec = dict(url='http://example.invalid', bytes=len(raw), sha256=expected,
                    body_views=[dict(domain='prose', view='lib_legacy',
                                     sha256=hashlib.sha256(body.encode()).hexdigest())])
        with fixture() as root, patch.object(datasets, 'CACHE', root):
            path = root / (key + '.raw')
            path.write_bytes(raw.replace(b'counter=1', b'counter=20'))
            datasets.verified_source.cache_clear()
            self.assertEqual(datasets.source_path(key, {key: spec}), path)
            self.assertTrue((root / (key+'.transport.json')).exists())
            path.write_bytes(path.read_bytes().replace('свет'.encode(), 'снег'.encode()))
            datasets.verified_source.cache_clear()
            with self.assertRaises(ValueError):
                datasets.source_path(key, {key: spec})

    def test_notebook_has_no_saved_outputs_or_remote_execution(self):
        notebook = json.loads((ROOT / 'inference.ipynb').read_text('utf8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        self.assertIn('from scripts.try_poetry import', code)
        self.assertIn('inference()', code)
        self.assertNotIn('ssh', code.lower())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])


if __name__ == '__main__':
    unittest.main()

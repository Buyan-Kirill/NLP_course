"""Report contracts: cumulative budgets, measured points and fair comparisons."""
import unittest

from reproduction.build_report import report_data
from reproduction.report_charts import ROOT, read_log, token_count


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = report_data()

    def test_pretrain_curve_preserves_both_logs_on_one_cumulative_axis(self):
        history = self.data['histories'][0]
        self.assertEqual(history['id'], 'pretrain')
        actual = history['curves']['perplexity'][0]['points']
        expected = []
        for source, offset in zip(history['sources'], [0, 630193557]):
            train, validation = read_log(ROOT/source)
            counts = {r['step']: token_count(r) for r in train}
            expected.extend((counts[r['step']]+offset, r['perplexity']) for r in validation)
        self.assertEqual([(p['x'], p['y']) for p in actual], expected)
        self.assertEqual(actual[-1]['x'], 680193557)
        self.assertTrue(all(a['x'] < b['x'] for a,b in zip(actual, actual[1:])))

    def test_stage_plots_keep_distinct_validation_sets(self):
        pretrain, poetry, sft = self.data['histories']
        self.assertEqual([r['id'] for r in [pretrain, poetry, sft]], ['pretrain','poetry','sft'])
        self.assertEqual(poetry['curves']['nll'][0]['points'][0]['y'],
                         self.data['poetry_metrics'][0]['nll'])
        endpoints = sft['curves']['nll'][1]['points']
        self.assertEqual([p['x'] for p in endpoints], [0,21680])
        self.assertEqual([p['y'] for p in endpoints], [r['nll'] for r in self.data['native']])

    def test_architecture_pairs_change_only_architecture_or_optimizer(self):
        rows = {r['label']:r for r in self.data['experiments']['architectures']}
        for left, right in [('A','C'),('B','D')]:
            self.assertEqual(rows[left]['config']['model'], rows[right]['config']['model'])
            self.assertEqual(rows[left]['config']['optimizer'], 'adamw')
            self.assertEqual(rows[right]['config']['optimizer'], 'muon')
            for field in ['data','tokenizer','seed','target_tokens','schedule_tokens','lr','min_lr']:
                self.assertEqual(rows[left]['config'][field], rows[right]['config'][field])
        self.assertEqual(rows['B']['config']['model'], rows['D']['config']['model'])


if __name__ == '__main__':
    unittest.main()

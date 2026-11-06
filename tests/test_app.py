import unittest
from app import fit, make_data, metrics, score


class AnomalyTests(unittest.TestCase):
    def test_reproducible_and_detects_injected_anomalies(self):
        self.assertEqual(make_data(4), make_data(4))
        model = fit(make_data(1, 300, 0))
        p, r, f1 = metrics(make_data(2, 200, 20), model, 4)
        self.assertGreater(p, 0.7)
        self.assertGreater(r, 0.7)
        self.assertGreater(f1, 0.7)

    def test_high_amount_has_higher_score(self):
        model = fit(make_data(3, 100, 0))
        self.assertGreater(score((500, 2, 1), model), score((30, 2, 0), model))


if __name__ == '__main__':
    unittest.main()

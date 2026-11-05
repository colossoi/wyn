"""Check fluid specialization capacities against the upstream allocation rule."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build import configuration


class ConfigurationTests(unittest.TestCase):
    def test_tree_reserve_matches_reference(self):
        # Reference max_leaves at n_crit=32, before sentinel/group padding.
        for particles, reference_leaves in [(1, 1024), (4320, 3141), (25000, 18181),
                                             (50000, 36363), (100000, 72727)]:
            with self.subTest(particles=particles):
                config = configuration(particles)
                capacity = config['tree_capacity']
                self.assertGreaterEqual(capacity - 1, reference_leaves)
                self.assertEqual((capacity - 2) % 7, 0)
        self.assertEqual(configuration()['tree_capacity'], 4097)

    def test_25000_particle_rebalance_fits(self):
        # Captured freeze: 6,231 leaves need six splits (seven extra leaves each).
        required_slots = 6231 + 6 * 7 + 1
        self.assertGreaterEqual(configuration(25000)['tree_capacity'], required_slots)

    def test_capacity_override_is_preserved(self):
        self.assertEqual(configuration(25000, tree_capacity=6260)['tree_capacity'], 6260)


if __name__ == '__main__':
    unittest.main()

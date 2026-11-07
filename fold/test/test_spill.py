"""The serial storage lowering must preserve distinct arrays and initialization."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spill import ARRAYS, spill


class StorageLowering(unittest.TestCase):
    def source(self):
        arrays='\n'.join(f'var a{i}: array<{kind}, {count}>; a{i}[0] = {kind}(0);'
                         for i,(kind,count) in enumerate(ARRAYS))
        return '@compute @workgroup_size(1, 1, 1) fn main() { loop { '+arrays+' break; } }'

    def test_each_allocation_stays_distinct_and_is_reinitialized_in_its_original_scope(self):
        lowered=spill(self.source())
        for i,(kind,count) in enumerate(ARRAYS):
            self.assertIn(f'a{i}: array<{kind}, {count}>',lowered)
            self.assertIn(f'fold_scratch.a{i}[0]',lowered)
            self.assertGreater(lowered.index(f'fold_zero_a{i} ='),lowered.index('loop {'))
        self.assertEqual(lowered.count('var<storage, read_write>'),1)

    def test_refuses_array_value_operations_instead_of_changing_alias_semantics(self):
        with self.assertRaisesRegex(ValueError,'Whole-array'):
            spill(self.source().replace('break;', 'let snapshot = a0; break;'))

    def test_refuses_parallel_invocations_or_unreviewed_layouts(self):
        for source in [self.source().replace('(1, 1, 1)','(64, 1, 1)'),
                       self.source().replace('13880','13881')]:
            with self.assertRaises(ValueError):spill(source)


if __name__=='__main__':unittest.main()

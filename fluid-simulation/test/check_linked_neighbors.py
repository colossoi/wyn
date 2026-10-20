"""Check Wyn linked nodes and cooperative traversal against geometric results."""
from pathlib import Path
import tempfile
from gpu import ROOT, run


def rows(values, width):
    return [values[i:i+width] for i in range(0, len(values), width)]


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='fluid-linked-') as tmp:
        for target in ['spirv', 'wgsl']:
            directory = Path(tmp)
            children, boxes, ranges, count = run(ROOT/'fluid-simulation/test/linked_tree.wyn', directory, target)
            assert count == [9], (target, count)
            assert children == [1]+[0]*16, (target, children)
            assert boxes[:4] == [0, 0, 0, 16], (target, boxes[:4])
            assert set(map(tuple, rows(boxes, 4)[1:9])) == {
                (x, y, z, 8) for x in [0, 8] for y in [0, 8] for z in [0, 8]
            }, (target, boxes)
            assert rows(ranges, 2)[1:9] == [[i, 1] for i in range(8)], (target, ranges)
            neighbors, = run(ROOT/'fluid-simulation/test/linked_neighbors.wyn', directory, target)
            for row, expected in zip(rows(neighbors, 98), [[0, 1], [0, 1], [2]], strict=True):
                assert row[96:] == [len(expected), 0], (target, row)
                assert row[:len(expected)] == expected, (target, row)
            print(target, 'linked nodes and root-only cooperative neighbors: passed', flush=True)
            positions, neighbors, count = run(ROOT/'fluid-simulation/test/linked_mixed_neighbors.wyn', directory, target)
            assert count == [17], (target, count)
            positions = rows(positions, 4)
            for i, row in enumerate(rows(neighbors, 98)):
                expected = {j for j, p in enumerate(positions)
                            if sum((positions[i][axis]-p[axis])**2 for axis in range(3)) < 25}
                assert row[96:] == [len(expected), 0], (target, i, row, expected)
                assert set(row[:row[96]]) == expected, (target, i, row, expected)
            print(target, 'mixed-depth cooperative neighbors match all-pairs oracle: passed', flush=True)

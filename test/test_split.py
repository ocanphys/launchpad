"""TagRanges: the tag of every position, from one entry per range."""

import unittest

import numpy as np

from artifacts.core.split import TagRanges


class TagRangesTests(unittest.TestCase):
    def test_a_slice_reads_each_positions_tag_across_range_edges(self):
        tags = TagRanges(np.array([0, 3, 4]), np.array([5, 9, 2]), length=7)
        self.assertEqual(tags[:].tolist(), [5, 5, 5, 9, 2, 2, 2])
        self.assertEqual(tags[2:5].tolist(), [5, 9, 2])
        self.assertEqual(tags[-2:99].tolist(), [2, 2])
        self.assertEqual(tags[4:4].tolist(), [])
        self.assertEqual(len(tags), 7)
        with self.assertRaises(ValueError):
            tags[::2]

    def test_untagged_is_tag_zero_everywhere(self):
        self.assertEqual(TagRanges.untagged(3)[:].tolist(), [0, 0, 0])
        self.assertEqual(TagRanges.untagged(0)[:].tolist(), [])

    def test_ranges_must_start_at_zero_and_increase_with_one_tag_each(self):
        for starts, tags in (([1, 3], [5, 9]), ([0, 3, 3], [5, 9, 2]), ([0, 3], [5]), ([], [])):
            with self.subTest(starts=starts, tags=tags), self.assertRaisesRegex(ValueError, "tag ranges"):
                TagRanges(np.array(starts, dtype=np.int64), np.array(tags, dtype=np.int64), length=7)


if __name__ == "__main__":
    unittest.main()

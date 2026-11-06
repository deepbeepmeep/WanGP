# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Windows mapped-page regression tests for WanGP #2573.

Run from the repository root: python -m unittest discover -s mmgp/tests -v
"""
import ctypes
import gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from mmgp import offload


@unittest.skipUnless(os.name == "nt", "Windows VirtualQuery regression")
class MappedPagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tensors = []
        self.addCleanup(self.release_tensors)

    def release_tensors(self):
        self.tensors.clear()
        gc.collect()

    def mapped_tensor(self, name="weights.bin"):
        path = Path(self.temp.name) / name
        path.write_bytes(bytes(3 * 4096))
        tensor = torch.from_file(str(path), shared=False, size=3 * 4096, dtype=torch.uint8)
        self.tensors.append(tensor)
        return tensor

    def test_empty_tensor_with_null_base_address_has_no_pages(self):
        info = offload._MemoryInformation()
        found = offload._virtual_query(0, ctypes.byref(info), ctypes.sizeof(info))
        self.assertNotEqual(found, 0)
        self.assertIsNone(info.BaseAddress)
        self.assertEqual(info.State, 0x10000)  # MEM_FREE
        tensor = torch.empty((1, 0, 1), dtype=torch.bfloat16)
        self.assertEqual(tensor.data_ptr(), 0)
        self.assertEqual(offload._mapped_pages([(tensor.data_ptr(), tensor.numel() * tensor.element_size())]), [])

    def test_zero_byte_spans_do_not_query_an_address(self):
        with patch.object(offload, "_virtual_query", side_effect=AssertionError("queried an empty span")):
            self.assertEqual(offload._mapped_pages([(0, 0), (0x12345000, 0)]), [])

    def test_empty_span_preserves_mapped_weights(self):
        tensor = self.mapped_tensor()
        base = tensor.data_ptr()
        self.assertEqual(offload._mapped_pages([(0, 0), (base, 1)]), [(base, base + 4096)])

    def test_mapped_ranges_round_merge_and_preserve_gaps(self):
        tensor = self.mapped_tensor()
        base = tensor.data_ptr()
        self.assertEqual(base % 4096, 0)
        self.assertEqual(offload._mapped_pages([(base + 4095, 2), (base + 2, 5)]), [(base, base + 8192)])
        self.assertEqual(offload._mapped_pages([(base + 8193, 1), (base + 1, 1)]),
                         [(base, base + 4096), (base + 8192, base + 12288)])

    def test_separate_file_views_remain_separate(self):
        first, second = self.mapped_tensor("a.bin"), self.mapped_tensor("b.bin")
        bases = sorted([first.data_ptr(), second.data_ptr()])
        self.assertEqual(offload._mapped_pages([(base, 12288) for base in bases]),
                         [(base, base + 12288) for base in bases])

    def test_private_memory_is_not_a_checkpoint_mapping(self):
        tensor = torch.ones(32)
        self.assertEqual(offload._mapped_pages([(tensor.data_ptr(), 128)]), [])

    def test_query_failure_has_no_mapping(self):
        with patch.object(offload, "_virtual_query", return_value=0):
            self.assertEqual(offload._mapped_pages([(0x12345000, 1)]), [])


if __name__ == "__main__":
    unittest.main()

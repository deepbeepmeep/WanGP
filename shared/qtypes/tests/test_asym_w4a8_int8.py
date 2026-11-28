# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import unittest

import torch

from shared.qtypes import asym_w4a8_int8 as w4a8


class W4A8ScaleTest(unittest.TestCase):
    def convert(self, scale, layer_format):
        weights = {
            "linear.weight": torch.full((1, 128), -1, dtype=torch.int8),
            "linear.weight_s_rel": scale,
            "linear.weight_s_channel": torch.ones(1),
        }
        metadata = None
        if layer_format:
            layers = {"linear": {"format": layer_format, "group_size": 16, "convrot_groupsize": 256}}
            metadata = {"_quantization_metadata": json.dumps({"layers": layers})}
        return w4a8.convert_to_quanto(weights, torch.float32, metadata=metadata)["state_dict"]

    def test_byte_backed_fp8(self):
        raw = torch.full((1, 16), 104, dtype=torch.uint8)
        weights = self.convert(raw, "asym_w4a8_int8")
        scale = weights["linear.weight._s_rel"]
        self.assertEqual(scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.data_ptr(), raw.data_ptr())
        self.assertTrue(torch.all(scale.float() == 64))
        codebook = torch.linspace(0, 1, 16)
        decoded = w4a8._decode_w4a8_torch(weights["linear.weight._data"], scale, codebook, 16)
        self.assertTrue(torch.all(decoded == 64))

    def test_other_scale_representations(self):
        for fmt, dtype in ((None, torch.uint8), ("w4a8_int8", torch.uint8),
                           ("asym_w4a8_int8", torch.int8), ("asym_w4a8_int8", torch.float32),
                           ("asym_w4a8_int8", torch.float8_e4m3fn)):
            with self.subTest(fmt=fmt, dtype=dtype):
                scale = torch.ones(1, 16).to(dtype)
                self.assertIs(self.convert(scale, fmt)["linear.weight._s_rel"], scale)

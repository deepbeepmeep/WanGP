import unittest
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.nn.attention import sdpa_kernel, SDPBackend
from bounded_attention import bounded_sdpa


class BoundedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        # A CPU generator avoids touching MPS seed/state.
        self.generator = torch.Generator(device="cpu").manual_seed(19)

    def check_attention(self, *, dtype=torch.float32, causal=False, mask_kind=None,
                        budget=208, query_count=19, key_count=13):
        tensors = [torch.randn(shape, generator=self.generator, device="cpu", dtype=dtype)
                   for shape in ((2, 4, query_count, 8), (2, 4, key_count, 8),
                                 (2, 4, key_count, 6))]
        mask = None
        if mask_kind == "bool":
            mask = torch.ones((query_count, key_count), dtype=torch.bool)
            mask[:, -2:] = False
            mask[0, :] = False
        elif mask_kind == "bias":
            mask = torch.randn((2, 4, query_count, key_count), generator=self.generator) * 0.1
        elif mask_kind == "broadcast":
            mask = torch.zeros((1, 1, 1, key_count))
            mask[..., -1] = float("-inf")
        calls, syncs = [], []
        def recording_sdpa(q, k, v, **kwargs):
            score_size = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * max(4, q.element_size())
            self.assertLessEqual(score_size, budget)
            calls.append(q.shape)
            return sdpa(q, k, v, **kwargs)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            expected = sdpa(*tensors, attn_mask=mask, is_causal=causal, scale=0.25)
            actual = bounded_sdpa(recording_sdpa, *tensors, attn_mask=mask,
                                  is_causal=causal, scale=0.25, score_budget_bytes=budget,
                                  synchronize=lambda: syncs.append(True))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(syncs), len(calls))
        self.assertEqual(actual.dtype, dtype)

    def test_query_chunks(self): self.check_attention()
    def test_head_chunks(self): self.check_attention(budget=2080)
    def test_boolean_mask_and_fully_masked_row(self): self.check_attention(mask_kind="bool")
    def test_per_head_additive_bias(self): self.check_attention(mask_kind="bias")
    def test_broadcast_mask(self): self.check_attention(mask_kind="broadcast")
    def test_causal_offset_and_longer_queries(self): self.check_attention(causal=True)
    def test_causal_offset_and_longer_keys(self): self.check_attention(causal=True, query_count=7, key_count=19)
    def test_bfloat16(self): self.check_attention(dtype=torch.bfloat16)
    def test_float16(self): self.check_attention(dtype=torch.float16)
    def test_float64(self): self.check_attention(dtype=torch.float64)
    def test_too_small_budget_fails_before_attention(self):
        t = torch.empty((1, 1, 4, 2))
        with self.assertRaises(ValueError):
            bounded_sdpa(lambda *a, **kw: self.fail("allocated attention"), t, t, t, score_budget_bytes=1)
    def test_attention_error_propagates_without_retry(self):
        t = torch.zeros((1, 1, 4, 2))
        calls = []
        def fail(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            bounded_sdpa(fail, t, t, t)
        self.assertEqual(len(calls), 1)
    def test_unsupported_layout_fails_before_attention(self):
        t = torch.empty((1, 1, 4, 2))
        with self.assertRaisesRegex(ValueError, "without GQA"):
            bounded_sdpa(lambda *a, **kw: self.fail("allocated attention"), t, t, t, enable_gqa=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

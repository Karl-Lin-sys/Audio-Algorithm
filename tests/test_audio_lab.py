import hashlib
import math
import tempfile
from pathlib import Path
import unittest

import torch
import torchaudio

from audio_lab import (audit_manifest, benchmark_cpu, mvdr_spectrum, read_wav,
                       sha256_file, si_sdr, summarize, write_wav)


class EvaluationTests(unittest.TestCase):
    def test_scale_invariance(self):
        generator = torch.Generator().manual_seed(8)
        ref = torch.randn(1000, generator=generator, dtype=torch.float64)
        est = ref + .1 * torch.randn(1000, generator=generator, dtype=torch.float64)
        self.assertAlmostEqual(si_sdr(est, ref), si_sdr(3 * est, ref), places=7)

    def test_undefined_scores_rejected(self):
        with self.assertRaises(ValueError):
            si_sdr(torch.ones(10), torch.zeros(10))
        with self.assertRaises(ValueError):
            si_sdr(torch.zeros(10), torch.arange(10.0))
        with self.assertRaises(ValueError):
            si_sdr(torch.zeros(9), torch.arange(10.0))
        with self.assertRaises(ValueError):
            si_sdr(torch.full((10,), float("nan")), torch.arange(10.0))

    def test_macro_exposes_majority_bias(self):
        rows = [{"condition": "easy", "delta_db": 10.0}] * 99
        rows += [{"condition": "wind", "delta_db": -10.0}]
        result = summarize(rows, ["easy", "wind"])
        self.assertAlmostEqual(result["micro_delta_db"], 9.8)
        self.assertEqual(result["macro_delta_db"], 0)
        self.assertEqual(result["worst_group_delta_db"], -10)

    def test_failure_or_missing_group_not_hidden(self):
        result = summarize([{"condition": "wind", "delta_db": None}], ["wind", "easy"])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["macro_delta_db"])
        self.assertEqual(result["groups"]["wind"]["failed"], 1)

    def test_each_leakage_axis(self):
        rows = [{"id": str(i), "split": split, "condition": "wind",
                 "sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                 **{key: f"{key}-{i}" for key in
                    ("speaker_id", "session_id", "source_id", "noise_source_id", "rir_id")}}
                for i, split in enumerate(("train", "val", "test"))]
        self.assertEqual(audit_manifest(rows)["counts"]["test"], 1)
        for key in ("speaker_id", "session_id", "source_id", "sha256", "noise_source_id", "rir_id"):
            with self.subTest(key=key):
                changed = [dict(row) for row in rows]
                changed[2][key] = changed[0][key]
                with self.assertRaises(ValueError):
                    audit_manifest(changed)

    def test_pcm_roundtrip_and_no_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.wav"
            x = torch.linspace(-.9, .9, 1600, dtype=torch.float64)
            write_wav(path, x, 16000)
            recovered, rate = read_wav(path)
            self.assertEqual(rate, 16000)
            self.assertLess(float((x - recovered[0]).abs().max()), 1 / 32768)
            self.assertEqual(len(sha256_file(path)), 64)
            with self.assertRaises(ValueError):
                read_wav(path, max_seconds=.01)
            with self.assertRaises(ValueError):
                write_wav(path, x * 2, 16000)

    def test_mvdr_rank_one_and_permutation(self):
        torch.manual_seed(5)
        speech = torch.randn(8, 64, dtype=torch.complex128)
        noise = torch.randn(8, 64, dtype=torch.complex128) * 2
        x = torch.stack((speech + noise, speech - noise))
        mask = speech.abs().square() / (speech.abs().square() + noise.abs().square())
        y = mvdr_spectrum(x, mask, 1-mask)
        swapped = mvdr_spectrum(x.flip(0), mask, 1-mask, reference_channel=1)
        torch.testing.assert_close(y, swapped, atol=1e-7, rtol=1e-7)
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(y.requires_grad)

    def test_zero_masks_fall_back_to_reference(self):
        x = torch.ones(2, 8, 16, dtype=torch.complex128)
        mask = torch.zeros(8, 16)
        torch.testing.assert_close(mvdr_spectrum(x, mask, mask), x[0])
        with self.assertRaises(ValueError):
            mvdr_spectrum(x, mask - 1, mask)

    def test_stft_inverse_preserves_length(self):
        x = torch.randn(2, 16017, dtype=torch.float64)
        spec = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=None).double()(x)
        y = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256).double()(spec, length=x.shape[-1])
        torch.testing.assert_close(x, y)

    def test_benchmark_counts_calls(self):
        calls = []
        result = benchmark_cpu(lambda: calls.append(1), 1, repeats=3, warmup=2)
        self.assertEqual(len(calls), 5)
        self.assertTrue(math.isfinite(result["median_rtf"]))


if __name__ == "__main__":
    unittest.main()

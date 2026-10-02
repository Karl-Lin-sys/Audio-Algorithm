"""Offline research utilities. No device, hearing-benefit or real-time claims.

API references and complexity notation are in docs/technical_report_ja.md.
All processing is CPU-only, one bounded-length recording at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import time
import wave

import numpy as np
import torch
import torchaudio


def read_wav(path: str | Path, max_seconds: float = 60.0):
    """Read PCM16 WAV as CPU float64 [C,N]. Time/space O(CN)."""
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    with wave.open(str(path), "rb") as src:
        rate, channels, count = src.getframerate(), src.getnchannels(), src.getnframes()
        if src.getsampwidth() != 2 or src.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM16 WAV is supported")
        if channels < 1 or count < 2 or count > rate * max_seconds:
            raise ValueError("Empty, invalid or too long WAV")
        raw = src.readframes(count)
    if len(raw) != count * channels * 2:
        raise ValueError("Truncated WAV")
    data = np.frombuffer(raw, dtype="<i2").reshape(count, channels).T.copy()
    return torch.from_numpy(data).to(torch.float64) / 32768.0, rate


def write_wav(path: str | Path, audio: torch.Tensor, rate: int):
    """Write mono PCM16, rejecting clipping rather than hiding it. O(N)."""
    if audio.ndim != 1 or audio.numel() < 2 or rate <= 0:
        raise ValueError("Expected nonempty mono waveform and positive rate")
    values = audio.detach().cpu().double()
    if not torch.isfinite(values).all() or values.abs().max() > 1:
        raise ValueError("Nonfinite or clipping output; inspect gain before export")
    pcm = (values * 32768).round().clamp(-32768, 32767).numpy().astype("<i2")
    with wave.open(str(path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(rate)
        dst.writeframes(pcm.tobytes())


@torch.inference_mode()
def si_sdr(estimate: torch.Tensor, reference: torch.Tensor):
    """Zero-mean, epsilon-regularized SI-SDR, dB; invalid silence is rejected. O(N)."""
    if estimate.ndim != 1 or estimate.shape != reference.shape or estimate.numel() < 2:
        raise ValueError("Aligned mono signals of identical length are required")
    estimate, reference = estimate.double(), reference.double()
    if not torch.isfinite(estimate).all() or not torch.isfinite(reference).all():
        raise ValueError("Nonfinite audio")
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    energy = reference.square().sum()
    if energy / reference.numel() <= 1e-12:
        raise ValueError("Silent/constant reference: SI-SDR is undefined")
    target = reference * (torch.dot(estimate, reference) / energy)
    # Relative epsilon gives an explicit finite floor/ceiling, not exact infinities.
    eps = energy * 1e-12
    if estimate.square().sum() / estimate.numel() <= 1e-12:
        raise ValueError("Silent/constant estimate: score separately as a failure")
    ratio = (target.square().sum() + eps) / ((estimate - target).square().sum() + eps)
    return float(10 * torch.log10(ratio))


@torch.inference_mode()
def mvdr_spectrum(spectrum: torch.Tensor, speech_mask: torch.Tensor,
                  noise_mask: torch.Tensor, reference_channel: int = 0):
    """Official PSD + SoudenMVDR. O(FTC^2 + FC^3), storage O(FTC^2) upper bound.

    Input complex [C,F,T], masks [F,T]. Whole-record covariance is NONCAUSAL.
    Masks must come from the mixture at deployment, never from test clean audio.
    """
    if spectrum.ndim != 3 or not spectrum.is_complex() or spectrum.device.type != "cpu":
        raise ValueError("Expected CPU complex [channels, frequencies, frames]")
    channels, bins, frames = spectrum.shape
    if min(channels, bins, frames) < 1 or not 0 <= reference_channel < channels:
        raise ValueError("Invalid spectrum/reference channel")
    if not torch.isfinite(spectrum).all():
        raise ValueError("Nonfinite spectrum")
    masks = []
    for mask in (speech_mask, noise_mask):
        if mask.shape != (bins, frames) or mask.is_complex() or mask.device.type != "cpu":
            raise ValueError("Expected CPU real [frequencies, frames] mask")
        if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError("Masks must be finite probabilities in [0,1]")
        masks.append(mask.to(spectrum.real.dtype))
    psd = torchaudio.transforms.PSD()
    speech_psd, noise_psd = (psd(spectrum, mask) for mask in masks)
    enhanced = torchaudio.transforms.SoudenMVDR()(
        spectrum, speech_psd, noise_psd, reference_channel,
        diagonal_loading=True, diag_eps=1e-6, eps=1e-8,
    )
    # Unsupported frequency bins cannot establish a reliable covariance estimate.
    valid = (masks[0].sum(-1) > 1e-8) & (masks[1].sum(-1) > 1e-8)
    enhanced = torch.where(valid[:, None], enhanced, spectrum[reference_channel])
    if not torch.isfinite(enhanced).all():
        raise ValueError("Nonfinite MVDR result; check covariance and microphone calibration")
    return enhanced


def audit_manifest(rows: list[dict]):
    """Reject cross-split identity/source/hash reuse. Expected O(NK), memory O(NK).

    IDs are dataset-global, not labels such as 'speaker_1' reused per dataset.
    This audits provided provenance, not unobserved pretraining contamination.
    """
    keys = ("speaker_id", "session_id", "source_id", "sha256", "noise_source_id", "rir_id")
    owners = {key: {} for key in keys}
    ids, counts = set(), {split: 0 for split in ("train", "val", "test")}
    if not rows:
        raise ValueError("Empty manifest")
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("id", "split", "condition", *keys)):
            raise ValueError("Missing/non-string provenance or condition")
        if row["split"] not in counts or row["id"] in ids:
            raise ValueError("Invalid split or duplicate sample id")
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        ids.add(row["id"])
        counts[row["split"]] += 1
        for key in keys:
            value = row[key]
            # Only optional acquisition components can be explicitly absent.
            if key in ("noise_source_id", "rir_id") and value == "not_applicable":
                continue
            previous = owners[key].setdefault(value, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Leakage: {key}={value} occurs in {previous} and {row['split']}")
    if not all(counts.values()):
        raise ValueError("All train/val/test splits must be present")
    return {"counts": counts, "status": "provided_metadata_passed",
            "limitation": "Does not detect near duplicates, false IDs, or pretraining overlap"}


def summarize(rows: list[dict], expected_conditions: list[str]):
    """Micro/macro/worst mean of paired improvements, recording failures. O(N+G).

    Rows have condition and delta_db; failed scores use delta_db=null.
    Macro is conditional on success. Any failure marks the report incomplete.
    """
    if not expected_conditions or len(set(expected_conditions)) != len(expected_conditions):
        raise ValueError("Declare unique target conditions before examining test results")
    groups = {g: {"count": 0, "valid": 0, "failed": 0, "sum": 0.0} for g in expected_conditions}
    for row in rows:
        if row.get("condition") not in groups:
            raise ValueError("Undeclared condition")
        group = groups[row["condition"]]
        group["count"] += 1
        value = row.get("delta_db")
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            group["failed"] += 1
        else:
            group["valid"] += 1
            group["sum"] += value
    means = []
    for group in groups.values():
        group["mean_delta_db"] = group["sum"] / group["valid"] if group["valid"] else None
        if group["valid"]:
            means.append(group["mean_delta_db"])
    valid = sum(g["valid"] for g in groups.values())
    complete = all(g["valid"] and not g["failed"] for g in groups.values())
    return {"groups": groups, "complete": complete,
            "micro_delta_db": sum(g["sum"] for g in groups.values()) / valid if valid else None,
            "macro_delta_db": statistics.mean(means) if len(means) == len(groups) else None,
            "worst_group_delta_db": min(means) if len(means) == len(groups) else None,
            "note": "Means condition on valid records; failures must gate acceptance"}


@torch.inference_mode()
def benchmark_cpu(operation, audio_seconds: float, repeats: int = 20, warmup: int = 3):
    """CPU only, sequential wall clock. O((R+W)Q + R log R), O(S+R) space."""
    if not math.isfinite(audio_seconds) or audio_seconds <= 0 or repeats < 2 or warmup < 0:
        raise ValueError("Invalid benchmark settings")
    for _ in range(warmup):
        operation()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()  # Discard outputs; no growing tensor/graph list.
        elapsed.append(time.perf_counter() - start)
    ordered = sorted(elapsed)
    return {"median_ms": statistics.median(elapsed) * 1000,
            "p95_ms": ordered[math.ceil(.95 * repeats) - 1] * 1000,
            "median_rtf": statistics.median(elapsed) / audio_seconds,
            "repeats": repeats, "warmup": warmup, "torch": torch.__version__,
            "torchaudio": torchaudio.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            "threads": torch.get_num_threads(), "device": "cpu",
            "scope": "offline whole-record kernel, excludes file I/O; not end-to-end latency"}


def sha256_file(path: str | Path):
    """Streaming file digest. O(B) time, O(1) auxiliary storage (1 MiB buffer)."""
    digest = hashlib.sha256()
    with open(path, "rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gtcrn(source_dir: str | Path, checkpoint: str | Path):
    """Load user-supplied official code and weights once. O(P+B), O(P) memory.

    This executes Python from source_dir: use only the documented official commit.
    No implicit network calls or arbitrary pickle objects are accepted.
    """
    source = Path(source_dir) / "gtcrn.py"
    spec = importlib.util.spec_from_file_location("official_gtcrn", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load gtcrn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GTCRN().cpu().eval()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights["model"], strict=True)
    return model


@torch.inference_mode()
def enhance_gtcrn(model, mono: torch.Tensor, sample_rate: int):
    """Official model with modern complex STFT API; offline only.

    Time O(T L log L + T A), space O(TF + T H + P); A = model ops/frame.
    """
    if sample_rate != 16000 or mono.ndim != 1 or mono.numel() <= 256:
        raise ValueError("GTCRN requires mono 16 kHz, longer than 256 samples")
    if mono.device.type != "cpu" or not torch.isfinite(mono).all():
        raise ValueError("Expected finite CPU waveform")
    model.eval()
    signal = mono.float()
    window = torch.hann_window(512).sqrt()
    spectrum = torch.stft(signal, 512, 256, 512, window, center=True, return_complex=True)
    output = model(torch.view_as_real(spectrum)[None])[0].contiguous()
    enhanced = torch.istft(torch.view_as_complex(output), 512, 256, 512, window,
                           center=True, length=signal.numel())
    if not torch.isfinite(enhanced).all():
        raise ValueError("Nonfinite model output")
    return enhanced


@torch.inference_mode()
def synthetic_demo():
    """Analytic two-microphone fixture with ORACLE masks; not speech evidence.

    O(FTC^2 + FC^3 + CT L log L), memory bounded by a fixed two-second fixture.
    """
    torch.manual_seed(17)
    rate, length = 16000, 32000
    t = torch.arange(length, dtype=torch.float64) / rate
    clean = .15 * torch.sin(2 * torch.pi * 440 * t) + .08 * torch.sin(2 * torch.pi * 880 * t)
    noise = .12 * torch.randn(length, dtype=torch.float64)
    target = torch.stack((clean, clean))
    interference = torch.stack((noise, -noise))
    mixture = target + interference
    stft = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=None).double()
    istft = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256).double()
    x, s, n = stft(mixture), stft(target), stft(interference)
    power_s, power_n = s[0].abs().square(), n[0].abs().square()
    speech_mask = power_s / (power_s + power_n).clamp_min(1e-15)
    noise_mask = power_n / (power_s + power_n).clamp_min(1e-15)
    def process():
        """One offline MVDR+ISTFT call; same asymptotics as mvdr_spectrum."""
        return istft(mvdr_spectrum(x, speech_mask, noise_mask), length=length)
    enhanced = process()
    return {"kind": "synthetic_oracle_unit_fixture_NOT_product_quality",
            "input_si_sdr_db": si_sdr(mixture[0], clean),
            "output_si_sdr_db": si_sdr(enhanced, clean),
            "benchmark": benchmark_cpu(process, length / rate),
            "limitations": ["oracle clean/noise masks", "stationary analytic tones",
                            "offline covariance", "not a trained or real-speech benchmark"]}


def main():
    """CLI dispatcher. Time/space is the selected operation plus JSON/file size."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    audit = sub.add_parser("audit")
    audit.add_argument("manifest")
    score = sub.add_parser("score")
    for name in ("reference", "mixture", "enhanced"):
        score.add_argument(name)
    summary = sub.add_parser("summarize")
    summary.add_argument("scores")
    summary.add_argument("--conditions", nargs="+", required=True)
    enhance = sub.add_parser("enhance")
    enhance.add_argument("input")
    enhance.add_argument("output")
    enhance.add_argument("--gtcrn-source", required=True)
    enhance.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        result = synthetic_demo()
    elif args.command == "audit":
        result = audit_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8-sig")))
    elif args.command == "summarize":
        result = summarize(json.loads(Path(args.scores).read_text(encoding="utf-8-sig")), args.conditions)
    elif args.command == "score":
        signals = [read_wav(getattr(args, name)) for name in ("reference", "mixture", "enhanced")]
        if len({rate for _, rate in signals}) != 1 or any(x.shape[0] != 1 for x, _ in signals):
            raise ValueError("Aligned mono audio at matching sample rates is required")
        reference, mixture, enhanced = [x[0] for x, _ in signals]
        before, after = si_sdr(mixture, reference), si_sdr(enhanced, reference)
        result = {"input_si_sdr_db": before, "output_si_sdr_db": after, "delta_db": after-before}
    else:
        if Path(args.input).resolve() == Path(args.output).resolve() or Path(args.output).exists():
            raise ValueError("Output must be a new path")
        audio, rate = read_wav(args.input)
        if audio.shape[0] != 1:
            raise ValueError("Explicitly select a reference microphone; no silent downmix")
        model = load_gtcrn(args.gtcrn_source, args.checkpoint)
        result_audio = enhance_gtcrn(model, audio[0], rate)
        write_wav(args.output, result_audio, rate)
        result = {"mode": "offline_gtcrn", "samples": result_audio.numel(), "rate": rate,
                  "source_sha256": sha256_file(Path(args.gtcrn_source) / "gtcrn.py"),
                  "checkpoint_sha256": sha256_file(args.checkpoint),
                  "input_sha256": sha256_file(args.input), "output_sha256": sha256_file(args.output)}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

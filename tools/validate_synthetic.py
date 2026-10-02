import hashlib
import json
from pathlib import Path
import sys

import argparse
parser = argparse.ArgumentParser(description='Synthetic reproducibility checks; no quality claims')
parser.add_argument('--gtcrn-source', required=True)
parser.add_argument('--output-dir', default='local_results')
args = parser.parse_args()
repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
import torch
import audio_lab as lab

torch.set_num_threads(1)
evidence = Path(args.output_dir)
evidence.mkdir(parents=True, exist_ok=True)
examples = evidence / 'examples'
examples.mkdir(exist_ok=True)

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

save(evidence / 'synthetic_oracle_demo.json', lab.synthetic_demo())
rows = []
for i, split in enumerate(('train', 'val', 'test')):
    rows.append({'id': f'example-{i}', 'split': split, 'condition': 'indoor',
                 'sha256': hashlib.sha256(f'fictional-{i}'.encode()).hexdigest(),
                 **{key: f'fictional-{key}-{i}' for key in
                    ('speaker_id', 'session_id', 'source_id', 'noise_source_id', 'rir_id')}})
save(examples / 'manifest.json', rows)
save(examples / 'scores.json', [{'condition': 'indoor', 'delta_db': 3.0},
                               {'condition': 'indoor', 'delta_db': 5.0},
                               {'condition': 'wind', 'delta_db': -2.0}])
save(evidence / 'manifest_example_result.json', lab.audit_manifest(rows))

source = Path(args.gtcrn_source)
checkpoint = source / 'checkpoints' / 'model_trained_on_dns3.tar'
model = lab.load_gtcrn(source, checkpoint)
torch.manual_seed(19)
t = torch.arange(16017) / 16000
x = .15 * torch.sin(2 * torch.pi * 240 * t) + .04 * torch.randn(t.numel())
enhanced = lab.enhance_gtcrn(model, x, 16000)
assert enhanced.shape == x.shape
assert torch.isfinite(enhanced).all()
assert not enhanced.requires_grad
assert not model.training
# Reference recipe expressed with supported complex STFT/ISTFT API.
with torch.inference_mode():
    window = torch.hann_window(512).sqrt()
    spectrum = torch.stft(x, 512, 256, 512, window, return_complex=True)
    predicted = model(torch.view_as_real(spectrum)[None])[0]
    reference = torch.istft(torch.view_as_complex(predicted.contiguous()),
                            512, 256, 512, window, length=x.numel())
torch.testing.assert_close(enhanced, reference)
# Real file CLI fixture uses our own synthetic waveform, not third-party recordings.
lab.write_wav(evidence / 'synthetic_input.wav', x, 16000)
result = {'kind': 'official_GTCRN_checkpoint_synthetic_smoke_test_NOT_quality_reproduction',
          'source_commit': '502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73',
          'source_sha256': lab.sha256_file(source / 'gtcrn.py'),
          'checkpoint_sha256': lab.sha256_file(checkpoint),
          'input_samples': x.numel(), 'output_samples': enhanced.numel(),
          'all_finite': True, 'grad_disabled': True, 'model_eval': True,
          'max_difference_from_reference_recipe': float((enhanced-reference).abs().max()),
          'total_parameters': sum(p.numel() for p in model.parameters()),
          'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
          'benchmark': lab.benchmark_cpu(lambda: lab.enhance_gtcrn(model, x, 16000), x.numel()/16000)}
save(evidence / 'gtcrn_smoke.json', result)
print(json.dumps(result, indent=2))

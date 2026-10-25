#!/usr/bin/env python3
"""Export JordiEncaboBataller/sign-language-cv checkpoints for browser inference.

Only run on a trusted checkout: PyTorch checkpoints contain pickled objects.
Build dependencies: torch, scikit-learn, numpy, onnx, onnxruntime.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path,
                        help='Local JordiEncaboBataller/sign-language-cv checkout')
    parser.add_argument('--output', type=Path, default=Path('assets'))
    args = parser.parse_args()
    source = args.source.resolve()

    # Load the exact upstream architectures without importing its camera/TTS UI.
    module = ast.parse((source / 'src/main_app.py').read_text())
    names = {'SignLanguageMLP', 'SignLanguageLSTM'}
    definitions = [node for node in module.body
                   if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in definitions} != names:
        parser.error('The checkout is missing the expected model architectures')
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'upstream_models', 'exec'), namespace)

    config = {'source': 'https://github.com/JordiEncaboBataller/sign-language-cv',
              'checkpoint_sha256': {}}
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for kind, filename, architecture in [
        ('static', 'mlp.pt', 'SignLanguageMLP'),
        ('motion', 'lstm_jz.pt', 'SignLanguageLSTM'),
    ]:
        checkpoint_path = source / 'saved_models' / filename
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = namespace[architecture](**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        classes = [str(label) for label in checkpoint['classes']]
        config[kind + '_classes'] = classes
        config['checkpoint_sha256'][filename] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

        if kind == 'static':
            if set(classes) != set('ABCDEFGHIKLMNOPQRSTUVWXY'):
                parser.error('Expected the 24 static ASL classes')
            scaler = checkpoint['scaler']
            config['scaler_mean'] = scaler.mean_.tolist()
            config['scaler_scale'] = scaler.scale_.tolist()
            shape = (1, 63)
            samples = np.load(source / 'data/processed/X_norm.npy', allow_pickle=False)
            samples = scaler.transform(samples[:32]).astype(np.float32)[:, None, :]
        else:
            if set(classes) != {'J', 'Z', 'STATIC'}:
                parser.error('Expected J, Z and STATIC motion classes')
            config['sequence_length'] = int(checkpoint['seq_len'])
            shape = (1, config['sequence_length'], 126)
            samples = np.load(source / 'data/processed/sequences_norm_v2.npy', allow_pickle=False)[:32, None]

        output_path = args.output / ('asl_' + kind + '.onnx')
        torch.onnx.export(model, torch.zeros(shape), str(output_path),
                          input_names=['landmarks'], output_names=['logits'],
                          opset_version=17, dynamo=False)
        session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        max_error = 0.0
        for sample in [*samples, *rng.normal(size=(8, *shape)).astype(np.float32)]:
            sample = np.asarray(sample, dtype=np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(sample)).numpy()
            actual = session.run(['logits'], {'landmarks': sample})[0]
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
            assert np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        print(f'{kind}: {len(samples) + 8} parity checks passed; max logit error {max_error:.3g}')

    (args.output / 'asl_model_config.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()

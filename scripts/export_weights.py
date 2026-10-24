"""One-time export of the repo's torch classifier to numpy (Q18).

    uv run --no-project -p 3.12 --with torch --with numpy \
        --index https://download.pytorch.org/whl/cpu \
        python scripts/export_weights.py .cache/asl-repo/ml/models/best_model.pth

Writes models/classifier.npz (BatchNorm folded into the Linear layers)
and tests/fixtures/parity.npz (inputs plus torch's softmax outputs).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent


def build_model():
    # Same layers as GestureModel.network in the repo's ml/model.py
    return nn.Sequential(
        nn.Linear(63, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 26),
    )


def main(pth):
    state = torch.load(pth, map_location="cpu")  # saved on cuda:0
    model = build_model()
    model.load_state_dict({k.removeprefix("network."): v for k, v in state.items()})
    model.eval()

    out = {}
    for i, (lin, bn) in enumerate([(0, 1), (4, 5), (8, 9), (12, None)]):
        w = model[lin].weight.detach().numpy().astype(np.float64)
        b = model[lin].bias.detach().numpy().astype(np.float64)
        if bn is not None:
            m = model[bn]
            scale = m.weight.detach().numpy() / np.sqrt(m.running_var.numpy() + m.eps)
            w = w * scale[:, None]
            b = (b - m.running_mean.numpy()) * scale + m.bias.detach().numpy()
        out[f"w{i}"] = w.T.astype(np.float32)  # stored (in, out) so inference is x @ w + b
        out[f"b{i}"] = b.astype(np.float32)
    (ROOT / "models").mkdir(exist_ok=True)
    np.savez(ROOT / "models" / "classifier.npz", **out)

    rng = np.random.default_rng(0)
    inputs = rng.random((32, 63), dtype=np.float32)
    inputs = inputs - inputs.min(axis=1, keepdims=True)
    inputs = inputs / inputs.max(axis=1, keepdims=True)  # same 0-1 range as real features
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(inputs)), dim=1).numpy()
    (ROOT / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / "tests" / "fixtures" / "parity.npz", inputs=inputs, probs=probs)
    print("wrote models/classifier.npz and tests/fixtures/parity.npz")


if __name__ == "__main__":
    main(sys.argv[1])

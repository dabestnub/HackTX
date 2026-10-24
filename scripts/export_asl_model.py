#!/usr/bin/env python3
"""Export the upstream SignSync classifier for local browser inference."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to a local asl-hand-gesture-recognition checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets"),
        help="Directory for the exported ONNX model and class labels",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    model_file = source / "ml" / "models" / "best_model.pth"
    classes_file = source / "ml" / "models" / "classes.pkl"
    if not model_file.is_file() or not classes_file.is_file():
        parser.error(f"{source} does not contain the expected trained model files")

    sys.path.insert(0, str(source))
    from ml.model import GestureModel

    classes = joblib.load(classes_file)
    model = GestureModel(len(classes))
    model.load_state_dict(
        torch.load(model_file, map_location="cpu", weights_only=True)
    )
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros((1, 63), dtype=torch.float32),
        str(args.output / "asl_classifier.onnx"),
        input_names=["landmarks"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    (args.output / "asl_classes.json").write_text(
        json.dumps(classes, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

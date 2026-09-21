from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from rl.checkpoints import model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "9x9" / "best_model.pth"
DEFAULT_OUTPUT_ROOT = ROOT / "frontend" / "public" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained GoNet checkpoint to ONNX for browser inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the source .pth checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to the exported .onnx file. Defaults to the board-size directory with v1/v2 filename inferred from inputs.",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=None,
        help="Override board size when it is missing from checkpoint metadata.",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Optional check: must match the width inferred from the checkpoint.",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=None,
        help="Optional check: must match the depth inferred from the checkpoint.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path for a JSON sidecar with model metadata.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, metadata, spec = model_from_checkpoint(checkpoint_path, board_size=args.board_size)
    board_size = spec['board_size']
    for key in ('num_channels', 'num_res_blocks'):
        override = getattr(args, key)
        if override is not None and override != spec[key]:
            raise ValueError(f"{key} does not match checkpoint")

    default_output_path = DEFAULT_OUTPUT_ROOT / f"{board_size}x{board_size}" / ("goplayer_v2.onnx" if spec["input_features"] == "pass-v2" else "goplayer_v1.onnx")
    output_path = resolve_path(args.output) if args.output is not None else default_output_path

    metadata_json_path = (
        resolve_path(args.metadata_json)
        if args.metadata_json
        else output_path.with_suffix(".json")
    )

    dummy_input = torch.zeros(1, spec['input_channels'], board_size, board_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            dynamo=False,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["board_state"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "board_state": {0: "batch"},
                "policy_logits": {0: "batch"},
                "value": {0: "batch"},
            },
        )
    onnx.checker.check_model(str(output_path))

    model_metadata = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT) if checkpoint_path.is_relative_to(ROOT) else checkpoint_path),
        "output": str(output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path),
        "board_size": board_size,
        "action_size": board_size * board_size + 1,
        "input_name": "board_state",
        "output_names": ["policy_logits", "value"],
        "num_channels": spec["num_channels"],
        "num_res_blocks": spec["num_res_blocks"],
        "input_features": spec["input_features"],
        "input_channels": spec["input_channels"],
        "komi": metadata.get("komi"),
        "opset": args.opset,
    }
    metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json_path.write_text(
        json.dumps(model_metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Exported ONNX model to {output_path}")
    print(f"Wrote metadata to {metadata_json_path}")
    print(json.dumps(model_metadata, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

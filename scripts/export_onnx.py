from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import onnx
import torch

from rl.net import GoNet
from rl.utils import load_checkpoint

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
        help="Optional path to the exported .onnx file. Defaults to frontend/public/models/{size}x{size}/goplayer_v1.onnx.",
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
        default=64,
        help="GoNet channel width. Must match training config.",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=3,
        help="GoNet residual block count. Must match training config.",
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


def parse_checkpoint_payload(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict):
        return payload, {}

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    state_dict = payload.get("model_state_dict") or payload.get("state_dict")
    if state_dict is None:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")
    return state_dict, metadata


def infer_board_size(state_dict: dict[str, Any]) -> int | None:
    policy_fc_weight = state_dict.get("policy_fc.weight")
    if policy_fc_weight is None or not hasattr(policy_fc_weight, "shape"):
        return None
    if len(policy_fc_weight.shape) != 2:
        return None

    action_size = int(policy_fc_weight.shape[0])
    flattened_features = int(policy_fc_weight.shape[1])
    board_area = action_size - 1
    if board_area <= 0:
        return None

    side = math.isqrt(board_area)
    if side * side != board_area:
        return None
    if flattened_features != 2 * side * side:
        return None
    return side


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict, metadata = parse_checkpoint_payload(payload)

    board_size = args.board_size
    if board_size is None:
        checkpoint_board_size = metadata.get("board_size")
        board_size = int(checkpoint_board_size) if checkpoint_board_size is not None else None
    if board_size is None:
        board_size = infer_board_size(state_dict)
    if board_size is None:
        raise ValueError("Unable to infer board size. Pass --board-size explicitly.")

    default_output_path = DEFAULT_OUTPUT_ROOT / f"{board_size}x{board_size}" / "goplayer_v1.onnx"
    output_path = resolve_path(args.output) if args.output is not None else default_output_path

    metadata_json_path = (
        resolve_path(args.metadata_json)
        if args.metadata_json
        else output_path.with_suffix(".json")
    )

    model = GoNet(
        size=board_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, board_size, board_size, dtype=torch.float32)
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
        "num_channels": args.num_channels,
        "num_res_blocks": args.num_res_blocks,
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

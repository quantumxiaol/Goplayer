"""Versioned input contracts shared by training, desktop inference and export."""
from .encoder import FEATURE_CHANNELS
from .net import GoNet
from .utils import load_checkpoint


def unpack_checkpoint(payload):
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    metadata = payload.get("metadata", {})
    channels = int(state["conv1.weight"].shape[1])
    inferred = {value: key for key, value in FEATURE_CHANNELS.items()}.get(channels)
    features = metadata.get("input_features", inferred)
    if features not in FEATURE_CHANNELS or FEATURE_CHANNELS[features] != channels:
        raise ValueError("Unsupported or inconsistent checkpoint input_features/input channels")
    area = int(state["policy_fc.weight"].shape[0]) - 1
    size = int(area ** 0.5)
    if size * size != area or metadata.get("board_size", size) != size:
        raise ValueError("Checkpoint board size is inconsistent")
    width = int(state["conv1.weight"].shape[0])
    blocks = len({key.split('.')[1] for key in state if key.startswith('res_blocks.')})
    spec = dict(board_size=size, num_channels=width, num_res_blocks=blocks,
                input_features=features, input_channels=channels)
    for key in ("num_channels", "num_res_blocks", "input_channels"):
        if metadata.get(key, spec[key]) != spec[key]:
            raise ValueError(f"Checkpoint {key} is inconsistent")
    return state, metadata, spec


def model_from_checkpoint(path, device="cpu", board_size=None, komi=None):
    state, metadata, spec = unpack_checkpoint(load_checkpoint(path, map_location="cpu"))
    if board_size is not None and spec["board_size"] != board_size:
        raise ValueError("Checkpoint board size does not match the game")
    if komi is not None and "komi" in metadata and metadata["komi"] != komi:
        raise ValueError("Checkpoint komi does not match the game")
    model = GoNet(spec["board_size"], spec["num_channels"], spec["num_res_blocks"], spec["input_features"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), metadata, spec

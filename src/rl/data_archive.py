"""Immutable per-game archives. NPZ contains only arrays (never Python pickle)."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .encoder import FEATURE_CHANNELS

FORMAT_VERSION = 1


def split_for_game(game_id, seed, validation_fraction):
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
    return "validation" if int.from_bytes(digest[:8], "big") / 2**64 < validation_fraction else "train"


def game_sgf(record):
    """SGF coordinates are column then row; empty coordinates mean Pass."""
    result = ""
    if record["winner"] == "draw":
        result = "RE[0]"
    elif record["winner"] in {"black", "white"}:
        result = f"RE[{'B' if record['winner'] == 'black' else 'W'}+{abs(record['score_diff']):g}]"
    root = (f"(;GM[1]FF[4]CA[UTF-8]AP[GoPlay]SZ[{record['board_size']}]"
            f"KM[{record['komi']:g}]{result}C[termination={record['termination']}; "
            "GoPlay area scoring, positional superko, no automatic dead-stone removal]")
    moves = []
    for row, col, color in record["move_sequence"]:
        coord = "" if (row, col) == (-1, -1) else chr(97 + col) + chr(97 + row)
        moves.append(f";{'B' if color == 'black' else 'W'}[{coord}]")
    return root + ''.join(moves) + ")\n"


def sample_arrays(samples, size, features):
    if not samples:
        return dict(states=np.empty((0, FEATURE_CHANNELS[features], size, size), dtype=np.uint8),
                    policies=np.empty((0, size * size + 1), dtype=np.float32),
                    values=np.empty(0, dtype=np.float32))
    states, policies, values = zip(*samples)
    return dict(states=torch.stack(list(states)).cpu().numpy().astype(np.uint8),
                policies=np.asarray(policies, dtype=np.float32), values=np.asarray(values, dtype=np.float32))


def read_samples(path, size, features):
    with np.load(path, allow_pickle=False) as archive:
        states, policies, values = (archive[key] for key in ("states", "policies", "values"))
    count = len(states)
    if (states.shape != (count, FEATURE_CHANNELS[features], size, size)
            or policies.shape != (count, size * size + 1) or values.shape != (count,)):
        raise ValueError(f"Invalid sample shapes: {path}")
    if (not np.isin(states, [0, 1]).all() or not np.isfinite(policies).all()
            or np.any(policies < 0) or not np.allclose(policies.sum(axis=1), 1, atol=1e-5)
            or not np.isin(values, [-1, 0, 1]).all()):
        raise ValueError(f"Invalid sample values: {path}")
    return [(torch.from_numpy(state.copy()).float(), policy.copy(), float(value))
            for state, policy, value in zip(states, policies, values)]


class GameArchive:
    def __init__(self, root, board_size, komi, input_features, seed, validation_fraction):
        self.root = Path(root)
        self.manifest = dict(format_version=FORMAT_VERSION, board_size=board_size, komi=komi,
                             input_features=input_features, split_seed=seed,
                             validation_fraction=validation_fraction)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "dataset.json").write_text(json.dumps(self.manifest, indent=2) + '\n')

    def save_game(self, iteration, game, samples, record):
        if record['board_size'] != self.manifest['board_size'] or record['komi'] != self.manifest['komi']:
            raise ValueError('Game does not match archive board size/komi')
        if record['termination'] != 'double_pass' and samples:
            raise ValueError('Truncated games cannot contain supervised samples')
        game_id = f"iter-{iteration:06d}-game-{game:04d}"
        destination = self.root / game_id
        if destination.exists():
            raise FileExistsError(destination)
        split = split_for_game(game_id, self.manifest['split_seed'], self.manifest['validation_fraction'])
        info = dict(record, game_id=game_id, iteration=iteration, game=game, split=split,
                    samples=len(samples), input_features=self.manifest['input_features'])
        # A complete directory is atomically published; incomplete writes are ignored by readers.
        with tempfile.TemporaryDirectory(prefix='.pending-', dir=self.root) as directory:
            temporary = Path(directory)
            (temporary / 'game.json').write_text(json.dumps(info, allow_nan=False) + '\n')
            (temporary / 'game.sgf').write_text(game_sgf(info))
            np.savez_compressed(temporary / 'samples.npz', **sample_arrays(
                samples, self.manifest['board_size'], self.manifest['input_features']))
            os.rename(temporary, destination)
        return split, game_id


def load_archive(root, train_buffer, validation_buffer, board_size, komi, input_features):
    """Rebuild bounded replay buffers in generation order; never re-split a game."""
    root = Path(root)
    manifest = json.loads((root / 'dataset.json').read_text())
    expected = dict(format_version=FORMAT_VERSION, board_size=board_size, komi=komi, input_features=input_features)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Dataset schema, board size, komi or input features do not match")
    index = []
    for directory in sorted(root.glob('iter-*-game-*')):
        record = json.loads((directory / 'game.json').read_text())
        if (record['game_id'] != directory.name or record['board_size'] != board_size
                or record['komi'] != komi or record['input_features'] != input_features
                or record['split'] != split_for_game(record['game_id'], manifest['split_seed'],
                                                     manifest['validation_fraction'])):
            raise ValueError(f'Game metadata/split does not match archive: {directory}')
        samples = read_samples(directory / 'samples.npz', board_size, input_features)
        if len(samples) != record['samples'] or (record['termination'] != 'double_pass' and samples):
            raise ValueError(f"Invalid sample count/termination: {directory}")
        if record['split'] not in {'train', 'validation'}:
            raise ValueError(f"Unknown split: {directory}")
        target = train_buffer if record['split'] == 'train' else validation_buffer
        target.save_game(samples)
        index.append(dict(game_id=record['game_id'], split=record['split'], samples=len(samples),
                          sha256=hashlib.sha256((directory / 'samples.npz').read_bytes()).hexdigest()))
    if not index:
        raise ValueError("Archive contains no complete games")
    return index

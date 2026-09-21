#!/usr/bin/env python3
"""AlphaZero-style training with auditable self-play and paired model selection."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Goplayer.goenv import GoEnv
from rl.checkpoints import unpack_checkpoint
from rl.data_archive import GameArchive, load_archive, game_sgf
from rl.encoder import FEATURE_CHANNELS
from rl.learning import LOSS_KEYS, update_model, validate_model
from rl.net import GoNet
from rl.replay_buffer import ReplayBuffer
from rl.selfplay import SearchConfig, evaluate_models, play_self_play_batch, should_promote
from rl.utils import load_checkpoint, resolve_device, save_checkpoint


BOARD_DEFAULTS = {
    9: dict(num_simulations=160, temperature_moves=40, min_moves_before_pass=50, max_moves=400),
    13: dict(num_simulations=320, temperature_moves=80, min_moves_before_pass=100, max_moves=676),
    19: dict(num_simulations=640, temperature_moves=120, min_moves_before_pass=180, max_moves=1444),
}
METRIC_FIELDS = [
    "iteration", "buffer", "games", "completed", "black_wins", "white_wins", "draws",
    "truncated", "truncation_rate", "black_score_rate", "mean_moves", "pass_rate",
    "mean_first_pass_ply", "mean_score_diff_completed", "mean_policy_entropy", "mean_policy_max",
    "mean_root_value_black", "mean_root_value_white", "selfplay_seconds", "positions_per_second",
    "optimizer_steps", *LOSS_KEYS, "validation_samples",
    *(f"val_{key}" for key in LOSS_KEYS),
    "reference_score", "champion_score", "promoted",
]


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_games(records):
    completed = [r for r in records if r["termination"] == "double_pass"]
    black = sum(r["winner"] == "black" for r in completed)
    white = sum(r["winner"] == "white" for r in completed)
    draws = sum(r["winner"] == "draw" for r in completed)
    moves = sum(r["moves"] for r in records)
    return {
        "games": len(records), "completed": len(completed), "black_wins": black,
        "white_wins": white, "draws": draws, "truncated": len(records) - len(completed),
        "truncation_rate": 1 - len(completed) / len(records) if records else 0.0,
        "black_score_rate": (black + 0.5 * draws) / len(completed) if completed else None,
        "mean_moves": mean_or_none([r["moves"] for r in records]),
        "pass_rate": sum(r["pass_count"] for r in records) / moves if moves else 0.0,
        "mean_first_pass_ply": mean_or_none([r["first_pass_ply"] for r in records]),
        "mean_score_diff_completed": mean_or_none([r["score_diff"] for r in completed]),
        "mean_policy_entropy": mean_or_none([r["policy_entropy"] for r in records]),
        "mean_policy_max": mean_or_none([r["policy_max"] for r in records]),
        "mean_root_value_black": mean_or_none([r["root_value_black"] for r in records]),
        "mean_root_value_white": mean_or_none([r["root_value_white"] for r in records]),
    }


def append_json(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def load_model_weights(model, path, board_size, komi):
    payload = load_checkpoint(path, map_location="cpu")
    state, metadata, spec = unpack_checkpoint(payload)
    if spec["input_features"] != model.input_features:
        raise ValueError("Input version mismatch: stones-v1 and pass-v2 weights cannot be interchanged")
    if spec["board_size"] != board_size:
        raise ValueError(f"Checkpoint board size does not match {board_size}: {path}")
    if "komi" in metadata and metadata["komi"] != komi:
        raise ValueError(f"Checkpoint komi does not match {komi}: {path}")
    model.load_state_dict(state, strict=True)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = resolve_device(args.device)
        self.config = SearchConfig(
            board_size=args.board_size, komi=args.komi, num_simulations=args.num_simulations,
            c_puct=args.c_puct, min_moves_before_pass=args.min_moves_before_pass,
            max_moves=args.max_moves, temperature_moves=args.temperature_moves,
            final_temperature=args.final_temperature, dirichlet_alpha=args.dirichlet_alpha,
            noise_fraction=args.noise_fraction,
        )
        self.rng = np.random.default_rng(args.seed)
        self.run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.checkpoint_dir = Path(args.checkpoint_dir) / f"{args.board_size}x{args.board_size}" / self.run_name
        self.log_dir = Path(args.log_dir) / f"{args.board_size}x{args.board_size}" / self.run_name
        if self.checkpoint_dir.exists() or self.log_dir.exists():
            raise FileExistsError("Run directory already exists; choose a new --run-name to avoid overwriting an experiment")
        self.model = GoNet(args.board_size, args.channels, args.res_blocks, args.input_features).to(self.device)
        if args.init_checkpoint:
            load_model_weights(self.model, args.init_checkpoint, args.board_size, args.komi)
        self.reference = copy.deepcopy(self.model).eval()
        if args.eval_opponent:
            load_model_weights(self.reference, args.eval_opponent, args.board_size, args.komi)
        # Champion starts as the initial model; the fixed reference never changes.
        self.champion = copy.deepcopy(self.model).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.buffer = ReplayBuffer(args.buffer_size)
        self.validation_buffer = ReplayBuffer(args.validation_buffer_size)
        if args.replay_dir:
            load_archive(args.replay_dir, self.buffer, self.validation_buffer,
                         args.board_size, args.komi, args.input_features)
        self.checkpoint_dir.mkdir(parents=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.archive = GameArchive(self.log_dir / "data", args.board_size, args.komi,
                                   args.input_features, args.seed, args.validation_fraction)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(run_name=self.run_name, resolved_device=str(self.device), torch_version=str(torch.__version__))
        try:
            config["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            config["git_commit"] = None
        (self.log_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        self.metrics_path = self.log_dir / "train_metrics.csv"
        with self.metrics_path.open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()
        self.writer = SummaryWriter(str(self.log_dir)) if args.tensorboard and SummaryWriter else None
        self.save_model("initial_model.pth", 0, self.model, selection="initial_weights")
        self.save_model("reference_model.pth", 0, self.reference, selection="fixed_reference")
        self.save_model("best_model.pth", 0, self.champion, selection="initial_baseline")

    def run(self):
        print(f"Training on {self.device}; checkpoints: {self.checkpoint_dir}; logs: {self.log_dir}", flush=True)
        try:
            for iteration in range(1, self.args.iterations + 1):
                started = time.perf_counter()
                records = []
                for start in range(0, self.args.games_per_iteration, self.args.self_play_batch_size):
                    count = min(self.args.self_play_batch_size, self.args.games_per_iteration - start)
                    for samples, record in play_self_play_batch(self.model, self.config, count, self.device, self.rng):
                        game = len(records) + 1
                        split, game_id = self.archive.save_game(iteration, game, samples, record)
                        target = self.buffer if split == "train" else self.validation_buffer
                        target.save_game(samples)
                        records.append(record)
                        append_json(self.log_dir / "selfplay_games.jsonl",
                                    dict(record, iteration=iteration, game=game, game_id=game_id, split=split))
                seconds = time.perf_counter() - started
                metrics = summarize_games(records)
                losses = []
                if len(self.buffer) >= self.args.batch_size:
                    for _ in range(self.args.train_steps_per_iteration):
                        losses.append(self.train_step())
                metrics.update({
                    "iteration": iteration, "buffer": len(self.buffer), "selfplay_seconds": seconds,
                    "positions_per_second": sum(r["moves"] for r in records) / max(seconds, 1e-9),
                    "optimizer_steps": len(losses),
                    **{key: mean_or_none([loss[key] for loss in losses]) for key in LOSS_KEYS},
                    **validate_model(self.model, list(self.validation_buffer.buffer), self.device, self.args.batch_size),
                    "reference_score": None, "champion_score": None, "promoted": False,
                })
                if iteration % self.args.eval_interval == 0:
                    evaluation_config = replace(self.config, num_simulations=self.args.eval_simulations)
                    evaluations = {}
                    for name, opponent in (("reference", self.reference), ("champion", self.champion)):
                        summary, games = evaluate_models(
                            self.model, opponent, evaluation_config, self.args.eval_games,
                            self.args.eval_opening_moves, self.args.seed + 10_000, self.device,
                            self.args.self_play_batch_size,
                        )
                        evaluations[name] = summary
                        metrics[f"{name}_score"] = summary["score"]
                        append_json(self.log_dir / "evaluation.jsonl", dict(iteration=iteration, opponent=name, **summary))
                        sgf_dir = self.log_dir / "evaluation_sgf"
                        sgf_dir.mkdir(exist_ok=True)
                        for number, game in enumerate(games, 1):
                            append_json(self.log_dir / "evaluation_games.jsonl",
                                        dict(game, iteration=iteration, opponent=name, game=number))
                            (sgf_dir / f"iter-{iteration:06d}-{name}-{number:04d}.sgf").write_text(game_sgf(game))
                    if should_promote(evaluations["champion"], self.args.promotion_threshold):
                        self.champion.load_state_dict(self.model.state_dict())
                        metrics["promoted"] = True
                        self.save_model("best_model.pth", iteration, self.champion,
                                        selection="paired_arena", evaluation=evaluations)
                    print(f"[eval={iteration}] {evaluations}; promoted={metrics['promoted']}", flush=True)
                if iteration % self.args.save_interval == 0 or iteration == self.args.iterations:
                    self.save_model(f"model_v{iteration}.pth", iteration, self.model, metrics=metrics)
                    self.save_model("latest_model.pth", iteration, self.model, metrics=metrics)
                with self.metrics_path.open("a", newline="") as stream:
                    csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writerow(metrics)
                if self.writer:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            self.writer.add_scalar(key, value, iteration)
                    self.writer.flush()
                print(f"[iter={iteration}] B/W/D={metrics['black_wins']}/{metrics['white_wins']}/{metrics['draws']} "
                      f"truncated={metrics['truncated']} buffer={len(self.buffer)} loss={metrics['total_loss']} "
                      f"positions/s={metrics['positions_per_second']:.2f}", flush=True)
                if metrics["truncation_rate"] > 0.25:
                    print("WARNING: >25% of games hit the move cap and were discarded; inspect games before extending training.", flush=True)
                if metrics["validation_samples"] == 0:
                    print("WARNING: No completed held-out games yet; validation metrics are empty.", flush=True)
        finally:
            if self.writer:
                self.writer.close()

    def train_step(self):
        samples = list(zip(*self.buffer.sample(self.args.batch_size)))
        return update_model(self.model, self.optimizer, samples, self.device, self.args.augment)

    def save_model(self, filename, iteration, model, **details):
        metadata = {
            "format_version": 3, "input_features": self.args.input_features,
            "input_channels": FEATURE_CHANNELS[self.args.input_features], "iteration": iteration, "board_size": self.args.board_size,
            "komi": self.args.komi, "num_channels": self.args.channels,
            "num_res_blocks": self.args.res_blocks, "run_name": self.run_name,
            "search_config": self.config.__dict__, **details,
        }
        save_checkpoint(self.checkpoint_dir / filename, model,
                        optimizer=self.optimizer if model is self.model else None, metadata=metadata)


def parse_args(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON defaults; explicit CLI options take precedence")
    parser.add_argument("--board-size", type=int, choices=BOARD_DEFAULTS, default=9)
    parser.add_argument("--input-features", choices=FEATURE_CHANNELS, default="stones-v1")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-buffer-size", type=int, default=20000)
    parser.add_argument("--replay-dir", type=Path, help="Import saved games into a NEW run; not an exact resume")
    parser.add_argument("--komi", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iteration", type=int, default=16)
    parser.add_argument("--self-play-batch-size", type=int, default=8)
    parser.add_argument("--train-steps-per-iteration", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=3)
    parser.add_argument("--temperature-moves", type=int, default=None)
    parser.add_argument("--final-temperature", type=float, default=0.25)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--min-moves-before-pass", type=int, default=os.getenv("RL_MIN_MOVES_BEFORE_PASS") or None)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-simulations", type=int, default=None)
    parser.add_argument("--eval-opening-moves", type=int, default=8)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--eval-opponent", type=Path, default=None, help="Optional fixed reference checkpoint")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Weights-only initialization of a NEW run; not an exact resume")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=os.getenv("RL_DEVICE", "auto"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(os.getenv("RL_CHECKPOINT_DIR", "checkpoints")))
    parser.add_argument("--log-dir", type=Path, default=Path(os.getenv("RL_LOG_DIR", "logs")))
    parser.add_argument("--run-name", default=os.getenv("RL_RUN_NAME") or None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction,
                        default=os.getenv("RL_TENSORBOARD", "0").lower() in {"1", "true", "yes"})
    parser.add_argument("--seed", type=int, default=42)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        try:
            defaults = json.loads(preliminary.config.read_text())
            allowed = {action.dest for action in parser._actions} - {"help", "config"}
            if not isinstance(defaults, dict) or set(defaults) - allowed:
                raise ValueError("Config must contain only recognized option names (use underscores)")
            parser.set_defaults(**defaults)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(argv)
    # argparse does not type-check non-string defaults from JSON; validate them.
    for action in parser._actions:
        if action.dest in {"help", "config"}:
            continue
        value = getattr(args, action.dest)
        if value is None:
            if action.dest not in {"komi", "num_simulations", "temperature_moves", "max_moves",
                                   "min_moves_before_pass", "dirichlet_alpha", "eval_simulations",
                                   "eval_opponent", "init_checkpoint", "run_name", "replay_dir"}:
                parser.error(f"{action.dest} cannot be null")
            continue
        if action.type is int and (isinstance(value, bool) or not isinstance(value, int)):
            parser.error(f"{action.dest} must be an integer")
        if action.type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                parser.error(f"{action.dest} must be a finite number")
        if isinstance(action, argparse.BooleanOptionalAction) and not isinstance(value, bool):
            parser.error(f"{action.dest} must be a boolean")
        if action.choices is not None and value not in action.choices:
            parser.error(f"Invalid {action.dest}: {value}")
        if action.type is Path:
            if not isinstance(value, (str, Path)):
                parser.error(f"{action.dest} must be a path")
            setattr(args, action.dest, Path(value))
    for key, value in BOARD_DEFAULTS[args.board_size].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.komi is None:
        args.komi = GoEnv._default_komi(args.board_size)
    if args.dirichlet_alpha is None:
        args.dirichlet_alpha = 10.0 / (args.board_size ** 2)
    if args.eval_simulations is None:
        args.eval_simulations = args.num_simulations
    positive = ["iterations", "games_per_iteration", "self_play_batch_size", "train_steps_per_iteration",
                "batch_size", "buffer_size", "validation_buffer_size", "num_simulations", "channels", "res_blocks", "max_moves",
                "save_interval", "eval_interval", "eval_games", "eval_simulations"]
    if any(getattr(args, key) <= 0 for key in positive):
        parser.error(f"These options must be positive: {', '.join(positive)}")
    if min(args.temperature_moves, args.min_moves_before_pass, args.eval_opening_moves) < 0:
        parser.error("Move thresholds must be nonnegative")
    if args.min_moves_before_pass + 2 > args.max_moves or args.eval_opening_moves >= args.max_moves:
        parser.error("max_moves must allow two Pass moves after the pass threshold and exceed the opening length")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation_fraction must be in (0, 1)")
    if args.buffer_size < args.batch_size:
        parser.error("buffer_size must be >= batch_size")
    if args.eval_games < 2 or args.eval_games % 2:
        parser.error("eval_games must be even and >= 2 for paired colors")
    if not 0 <= args.noise_fraction <= 1 or not 0.5 < args.promotion_threshold <= 1:
        parser.error("noise_fraction must be in [0, 1]; promotion_threshold in (0.5, 1]")
    if min(args.dirichlet_alpha, args.c_puct, args.learning_rate) <= 0 or args.final_temperature < 0:
        parser.error("alpha, c_puct and learning_rate must be positive; temperature nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if args.run_name is not None and (not isinstance(args.run_name, str) or args.run_name in {".", ".."}
                          or "/" in args.run_name or "\\" in args.run_name):
        parser.error("run_name must be a single directory name")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    Trainer(args).run()


if __name__ == "__main__":
    main()

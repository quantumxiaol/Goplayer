#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on optional rl deps
    raise ImportError("NumPy is required. Install with: uv sync --extra rl") from exc

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - depends on optional rl deps
    raise ImportError("PyTorch is required. Install with: uv sync --extra rl") from exc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Goplayer.goenv import GoEnv
from rl.encoder import encode_state
from rl.mcts import MCTS, PASS_MOVE, index_to_action, other_color
from rl.net import GoNet
from rl.replay_buffer import ReplayBuffer
from rl.utils import ensure_checkpoint_dir, resolve_device, save_checkpoint


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = resolve_device(args.device)
        self.checkpoint_dir = ensure_checkpoint_dir(args.checkpoint_dir)
        self.log_dir = ensure_checkpoint_dir(args.log_dir)
        self.buffer = ReplayBuffer(capacity=args.buffer_size)

        self.model = GoNet(
            size=args.board_size,
            num_channels=args.channels,
            num_res_blocks=args.res_blocks,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.best_loss = float("inf")

        self.metrics_path = self.log_dir / "train_metrics.csv"
        if not self.metrics_path.exists():
            self.metrics_path.write_text(
                "iteration,buffer,black_wins,white_wins,black_win_rate,total_loss,policy_loss,value_loss\n",
                encoding="utf-8",
            )

        self.writer = None
        if args.tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def run(self):
        print(f"Training on device: {self.device}")
        for iteration in range(1, self.args.iterations + 1):
            wins = {"black": 0, "white": 0}

            for _ in range(self.args.games_per_iteration):
                game_samples, winner = self.self_play_game()
                self.buffer.save_game(game_samples)
                wins[winner] += 1

            losses = []
            if len(self.buffer) >= self.args.batch_size:
                for _ in range(self.args.train_steps_per_iteration):
                    losses.append(self.train_step())

            avg_total_loss = float(np.mean([item["total"] for item in losses])) if losses else float("inf")
            avg_policy_loss = float(np.mean([item["policy"] for item in losses])) if losses else float("inf")
            avg_value_loss = float(np.mean([item["value"] for item in losses])) if losses else float("inf")
            black_win_rate = wins["black"] / max(self.args.games_per_iteration, 1)

            print(
                f"[iter={iteration}] buffer={len(self.buffer)} "
                f"wins(B/W)=({wins['black']}/{wins['white']}) "
                f"loss={avg_total_loss:.4f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("win_rate/black", black_win_rate, iteration)
                if losses:
                    self.writer.add_scalar("loss/total", avg_total_loss, iteration)
                    self.writer.add_scalar("loss/policy", avg_policy_loss, iteration)
                    self.writer.add_scalar("loss/value", avg_value_loss, iteration)

            with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(
                    f"{iteration},{len(self.buffer)},{wins['black']},{wins['white']},{black_win_rate:.6f},"
                    f"{avg_total_loss:.6f},{avg_policy_loss:.6f},{avg_value_loss:.6f}\n"
                )

            if iteration % self.args.save_interval == 0:
                self.save_model(f"model_v{iteration}.pth", iteration, avg_total_loss, wins)
                if avg_total_loss < self.best_loss:
                    self.best_loss = avg_total_loss
                    self.save_model("best_model.pth", iteration, avg_total_loss, wins)

        if self.writer is not None:
            self.writer.close()

    def self_play_game(self):
        env = GoEnv(size=self.args.board_size)
        current_color = "black"
        move_count = 0
        game_history = []
        mcts = MCTS(self.model, c_puct=self.args.c_puct, num_simulations=self.args.num_simulations)

        while not env.game_over and move_count < self.args.max_moves:
            state = encode_state(env, current_color)
            temperature = 1.0 if move_count < self.args.temperature_moves else 1e-3
            action_probs = mcts.get_action_probs(env, current_color, temperature=temperature, device=self.device)
            action_index = int(np.random.choice(len(action_probs), p=action_probs))
            action = index_to_action(action_index, env.size)

            game_history.append((state, action_probs, current_color))

            if action == PASS_MOVE:
                env.register_pass()
            else:
                row, col = action
                if not env.place_stone(row, col, current_color):
                    env.register_pass()

            current_color = other_color(current_color)
            move_count += 1

        result = env.judge_winner()
        winner = result["winner"]
        processed_samples = []
        for state, target_policy, player in game_history:
            value = 1.0 if player == winner else -1.0
            processed_samples.append((state, target_policy, value))
        return processed_samples, winner

    def train_step(self):
        states, policies, values = self.buffer.sample(self.args.batch_size)
        if not states:
            return {"total": float("inf"), "policy": float("inf"), "value": float("inf")}

        state_batch = torch.stack([state.float() for state in states]).to(self.device)
        policy_batch = torch.tensor(np.stack(policies), dtype=torch.float32, device=self.device)
        value_batch = torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.model.train()
        policy_logits, value_preds = self.model(state_batch)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_batch * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value_preds, value_batch)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "total": float(total_loss.item()),
            "policy": float(policy_loss.item()),
            "value": float(value_loss.item()),
        }

    def save_model(self, filename: str, iteration: int, avg_loss: float, wins):
        path = self.checkpoint_dir / filename
        metadata = {
            "iteration": iteration,
            "board_size": self.args.board_size,
            "avg_total_loss": avg_loss,
            "wins": wins,
        }
        save_checkpoint(path, self.model, optimizer=self.optimizer, metadata=metadata)


def parse_args():
    load_dotenv()

    env_device = os.getenv("RL_DEVICE", "auto")
    env_checkpoint_dir = os.getenv("RL_CHECKPOINT_DIR")
    env_log_dir = os.getenv("RL_LOG_DIR")
    env_tensorboard = _env_bool("RL_TENSORBOARD", False)

    parser = argparse.ArgumentParser(description="AlphaZero-like self-play training for Goplayer")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--train-steps-per-iteration", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--num-simulations", type=int, default=80)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=3)
    parser.add_argument("--temperature-moves", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=400)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--device", type=str, default=env_device)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(env_checkpoint_dir) if env_checkpoint_dir else ROOT / "checkpoints",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=env_tensorboard,
        help="Enable or disable TensorBoard logging.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(env_log_dir) if env_log_dir else ROOT / "logs",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()

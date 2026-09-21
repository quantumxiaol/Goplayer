#!/usr/bin/env python3
"""Compare update budgets on ONE frozen dataset, with no self-play generation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from rl.checkpoints import model_from_checkpoint
from rl.data_archive import load_archive, game_sgf
from rl.learning import LOSS_KEYS, update_model, validate_model
from rl.replay_buffer import ReplayBuffer
from rl.selfplay import SearchConfig, evaluate_models
from rl.utils import resolve_device, save_checkpoint


def run_comparison(args):
    device = resolve_device(args.device)
    initial, metadata, spec = model_from_checkpoint(args.checkpoint, device)
    if 'komi' not in metadata or 'search_config' not in metadata:
        raise ValueError('Use an initial_model.pth from an archived run (komi and search_config required)')
    train, validation = ReplayBuffer(args.buffer_size), ReplayBuffer(args.validation_buffer_size)
    index = load_archive(args.data_dir, train, validation, spec['board_size'],
                         metadata['komi'], spec['input_features'])
    if len(train) < args.batch_size or not len(validation):
        raise ValueError('Need at least batch_size training samples and nonempty held-out games')
    search_config = SearchConfig(**metadata['search_config'])
    if args.opening_moves >= search_config.max_moves:
        raise ValueError('opening_moves must be less than the source search max_moves')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Snapshot membership/checksums so both branches can be audited after source files change.
    dataset_id = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
    manifest = dict(arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    dataset_sha256=dataset_id, games=index, train_samples=len(train), validation_samples=len(validation),
                    initial_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    model=spec, search_config=metadata['search_config'],
                    optimizer='fresh Adam, weight_decay=1e-4', torch_version=str(torch.__version__))
    (args.output_dir / 'comparison.json').write_text(json.dumps(manifest, indent=2) + '\n')
    train_samples, held_out = list(train.buffer), list(validation.buffer)
    fields = ['round', 'updates', *LOSS_KEYS, 'validation_samples', *(f'val_{k}' for k in LOSS_KEYS)]
    summaries = []
    for budget in args.updates:
        # Same weights, fresh optimizer, and the same sampling/augmentation stream prefix.
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        sample_rng, augment_rng = random.Random(args.seed), random.Random(args.seed + 1)
        torch.manual_seed(args.seed)
        branch = args.output_dir / f'updates-{budget}'
        branch.mkdir()
        with (branch / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(round=0, updates=0, **validate_model(model, held_out, device, args.batch_size)))
            for iteration in range(1, args.rounds + 1):
                losses = [update_model(model, optimizer, sample_rng.sample(train_samples, args.batch_size),
                                       device, augment=args.augment, rng=augment_rng) for _ in range(budget)]
                row = dict(round=iteration, updates=iteration * budget,
                           **{key: sum(loss[key] for loss in losses) / len(losses) for key in LOSS_KEYS},
                           **validate_model(model, held_out, device, args.batch_size))
                writer.writerow(row)
                stream.flush()
                print(f"[updates/round={budget}] {row}", flush=True)
        details = dict(metadata, **spec, comparison_dataset=dataset_id, optimizer_steps=args.rounds * budget,
                       learning_rate=args.learning_rate, selection='offline_comparison')
        save_checkpoint(branch / 'model.pth', model, optimizer, details)
        result = dict(updates_per_round=budget, **row)
        if not args.skip_arena:
            summary, games = evaluate_models(model, initial, search_config, args.eval_games,
                                            args.opening_moves, args.seed + 10_000, device, args.eval_batch_size)
            result['arena_vs_initial'] = summary
            with (branch / 'evaluation_games.jsonl').open('w') as stream:
                for number, game in enumerate(games, 1):
                    stream.write(json.dumps(game) + '\n')
                    (branch / f'eval-{number:04d}.sgf').write_text(game_sgf(game))
        summaries.append(result)
        (args.output_dir / 'results.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--updates', type=int, nargs='+', default=[20, 80])
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--validation-buffer-size', type=int, default=20000)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='cuda')
    parser.add_argument('--skip-arena', action='store_true', help='Only loss diagnostics; no claim about playing strength')
    parser.add_argument('--eval-games', type=int, default=20)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--opening-moves', type=int, default=8)
    args = parser.parse_args(argv)
    if (min(args.updates + [args.rounds, args.batch_size, args.buffer_size, args.validation_buffer_size,
                            args.eval_games, args.eval_batch_size]) <= 0 or len(set(args.updates)) != len(args.updates)
            or args.eval_games % 2 or args.buffer_size < args.batch_size or args.opening_moves < 0
            or not 0 <= args.seed < 2**32 or not 0 < args.learning_rate < float('inf')):
        parser.error('Invalid budget, batch size, seed, learning rate or paired evaluation settings')
    return args


if __name__ == '__main__':
    run_comparison(parse_args())

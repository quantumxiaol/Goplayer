#!/usr/bin/env python3
"""Paired, noise-free evaluation of two saved models across opening seeds; no training."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from rl.checkpoints import model_from_checkpoint
from rl.data_archive import game_sgf
from rl.selfplay import SearchConfig, evaluate_models, summarize_evaluation
from rl.utils import resolve_device


def run_evaluation(args):
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory; existing results will not be overwritten')
    device = resolve_device(args.device)
    candidate, metadata, spec = model_from_checkpoint(args.candidate, device)
    if 'komi' not in metadata or 'search_config' not in metadata:
        raise ValueError('Candidate checkpoint must include komi and search_config metadata')
    config = SearchConfig(**metadata['search_config'])
    if config.board_size != spec['board_size'] or config.komi != metadata['komi']:
        raise ValueError('Candidate search_config disagrees with checkpoint board size/komi')
    opponent, opponent_metadata, opponent_spec = model_from_checkpoint(
        args.opponent, device, board_size=config.board_size, komi=config.komi,
    )
    if 'komi' not in opponent_metadata:
        raise ValueError('Opponent checkpoint must include komi metadata')
    if args.num_simulations is not None:
        config = replace(config, num_simulations=args.num_simulations)
    if (config.num_simulations < 1 or not math.isfinite(config.c_puct) or config.c_puct <= 0
            or not math.isfinite(config.komi) or config.min_moves_before_pass < 0
            or config.max_moves < config.min_moves_before_pass + 2
            or args.opening_moves >= config.max_moves):
        raise ValueError('Invalid shared search configuration or opening length')
    args.output_dir.mkdir(parents=True)
    manifest = dict(
        candidate=str(args.candidate), opponent=str(args.opponent),
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        opponent_sha256=hashlib.sha256(args.opponent.read_bytes()).hexdigest(),
        candidate_spec=spec, opponent_spec=opponent_spec, search_config=asdict(config),
        seeds=args.seeds, games_per_seed=args.games_per_seed, opening_moves=args.opening_moves,
        batch_size=args.batch_size, device=str(device), add_root_noise=False,
        move_selection='argmax', score_perspective='candidate',
    )
    (args.output_dir / 'evaluation_config.json').write_text(json.dumps(manifest, indent=2) + '\n')
    all_games, per_seed = [], []
    for seed in args.seeds:
        print(f'[seed={seed}] starting {args.games_per_seed} games; candidate plays both colors', flush=True)
        summary, games = evaluate_models(candidate, opponent, config, args.games_per_seed,
                                        args.opening_moves, seed, device, args.batch_size)
        per_seed.append(dict(seed=seed, **summary))
        with (args.output_dir / 'games.jsonl').open('a') as stream:
            for number, game in enumerate(games, 1):
                stream.write(json.dumps(dict(game, seed=seed, game=number), allow_nan=False) + '\n')
                (args.output_dir / f'seed-{seed}-game-{number:04d}.sgf').write_text(game_sgf(game))
        all_games.extend(games)
        result = dict(candidate=str(args.candidate), opponent=str(args.opponent),
                      score_perspective='candidate', completed_seeds=len(per_seed),
                      planned_seeds=len(args.seeds), per_seed=per_seed,
                      overall=summarize_evaluation(all_games))
        temporary = args.output_dir / 'results.json.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output_dir / 'results.json')
        print(f'[seed={seed}] {summary}', flush=True)
    print(f"[overall, candidate perspective] {result['overall']}", flush=True)
    if result['overall']['truncated']:
        print('WARNING: Unfinished games are excluded from scores; inspect truncation before comparing strength.', flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--opponent', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261, 20262])
    parser.add_argument('--games-per-seed', type=int, default=20,
                        help='Total games for each seed, including both colors; must be even')
    parser.add_argument('--opening-moves', type=int, default=8)
    parser.add_argument('--num-simulations', type=int, help='Shared search budget; defaults to candidate metadata')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='cuda')
    args = parser.parse_args(argv)
    if (args.games_per_seed < 2 or args.games_per_seed % 2 or args.batch_size < 1
            or args.opening_moves < 1 or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**32 for seed in args.seeds)
            or (args.num_simulations is not None and args.num_simulations < 1)):
        parser.error('Require even games >= 2, positive opening/batch/search sizes and distinct seeds in [0, 2**32)')
    return args


if __name__ == '__main__':
    run_evaluation(parse_args())

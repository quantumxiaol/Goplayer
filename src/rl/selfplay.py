"""Self-play and paired evaluation; no optimizer or training loop lives here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Goplayer.goenv import GoEnv
from .encoder import encode_state
from .mcts import MCTS, PASS_MOVE, index_to_action, other_color, temperature_policy


@dataclass(frozen=True)
class SearchConfig:
    board_size: int
    komi: float
    num_simulations: int
    c_puct: float
    min_moves_before_pass: int
    max_moves: int
    temperature_moves: int
    final_temperature: float
    dirichlet_alpha: float
    noise_fraction: float


def apply_action(env, color, action):
    if action == PASS_MOVE:
        env.register_pass(color)
    elif not env.place_stone(*action, color):
        raise RuntimeError(f"Search returned an illegal action: {color} {action}")


def outcome_record(env):
    black, white = env.calculate_area_score()
    winner = None
    if env.game_over:
        winner = "draw" if black == white else ("black" if black > white else "white")
    passes = [i + 1 for i, (r, c, _) in enumerate(env.moves_history) if (r, c) == PASS_MOVE]
    return {
        "termination": "double_pass" if env.game_over else "max_moves",
        "winner": winner,
        "moves": len(env.moves_history),
        "pass_count": len(passes),
        "first_pass_ply": passes[0] if passes else None,
        "black_score": black,
        "white_score": white,
        "score_diff": black - white,  # A position estimate only when truncated.
        "board_size": env.size,
        "komi": env.komi,
        "move_sequence": [list(move) for move in env.moves_history],
        "final_board": [row.copy() for row in env.grid],
    }


def finish_samples(history, record):
    # A move cap is not a terminal result. Discard rather than fabricate labels.
    if record["termination"] != "double_pass":
        return []
    winner = record["winner"]
    return [(state, policy, 0.0 if winner == "draw" else (1.0 if color == winner else -1.0))
            for state, policy, color in history]


def play_self_play_batch(model, config, count, device, rng):
    envs = [GoEnv(config.board_size, config.komi, record_history=False) for _ in range(count)]
    colors = ["black"] * count
    histories = [[] for _ in envs]
    entropies, maxima = [[] for _ in envs], [[] for _ in envs]
    values = [{"black": [], "white": []} for _ in envs]
    search = MCTS(model, config.c_puct, config.num_simulations)
    while True:
        active = [i for i, env in enumerate(envs) if not env.game_over and len(env.moves_history) < config.max_moves]
        if not active:
            break
        policies = search.get_action_probs_batch(
            [envs[i] for i in active], [colors[i] for i in active], device=device,
            temperature=1.0, add_root_noise=True, dirichlet_alpha=config.dirichlet_alpha,
            noise_fraction=config.noise_fraction, rng=rng,
            min_moves_before_pass=config.min_moves_before_pass,
        )
        for i, policy, root_value in zip(active, policies, search.last_root_values):
            env, color = envs[i], colors[i]
            histories[i].append((encode_state(env, color, getattr(model, "input_features", "stones-v1")), policy.copy(), color))
            positive = policy[policy > 0]
            entropies[i].append(float(-(positive * np.log(positive)).sum()))
            maxima[i].append(float(policy.max()))
            values[i][color].append(root_value)
            temperature = 1.0 if len(env.moves_history) < config.temperature_moves else config.final_temperature
            sampling_policy = temperature_policy(policy, temperature)
            # Normalize in float64 for numpy.choice's stricter probability check.
            sampling_policy = sampling_policy.astype(np.float64)
            sampling_policy /= sampling_policy.sum()
            action = index_to_action(int(rng.choice(len(policy), p=sampling_policy)), env.size)
            apply_action(env, color, action)
            colors[i] = other_color(color)
    results = []
    for i, env in enumerate(envs):
        record = outcome_record(env)
        record.update({
            "policy_entropy": float(np.mean(entropies[i])) if entropies[i] else 0.0,
            "policy_max": float(np.mean(maxima[i])) if maxima[i] else 0.0,
            "root_value_black": float(np.mean(values[i]["black"])) if values[i]["black"] else None,
            "root_value_white": float(np.mean(values[i]["white"])) if values[i]["white"] else None,
        })
        results.append((finish_samples(histories[i], record), record))
    return results


def paired_openings(config, games, opening_moves, seed):
    if games < 2 or games % 2:
        raise ValueError("Evaluation games must be positive and even")
    rng = np.random.default_rng(seed)
    envs, candidate_colors, to_play = [], [], []
    for _ in range(games // 2):
        env = GoEnv(config.board_size, config.komi, record_history=False)
        color = "black"
        for _ in range(opening_moves):
            legal = env.legal_moves(color)
            if not legal:
                break
            apply_action(env, color, legal[int(rng.integers(len(legal)))])
            color = other_color(color)
        # Same opening, candidate plays both colors; seed is fixed across rounds.
        envs.extend([env.clone_for_search(), env.clone_for_search()])
        candidate_colors.extend(["black", "white"])
        to_play.extend([color, color])
    return envs, candidate_colors, to_play


def summarize_evaluation(records):
    wins = sum(r["winner"] == r["candidate_color"] for r in records)
    draws = sum(r["winner"] == "draw" for r in records)
    truncated = sum(r["winner"] is None for r in records)
    completed = len(records) - truncated
    result = {"games": len(records), "completed": completed, "truncated": truncated,
              "wins": wins, "draws": draws, "losses": completed - wins - draws,
              "score": (wins + 0.5 * draws) / completed if completed else None}
    for color in ("black", "white"):
        subset = [r for r in records if r["candidate_color"] == color and r["winner"] is not None]
        points = sum(1 if r["winner"] == color else 0.5 if r["winner"] == "draw" else 0 for r in subset)
        result[f"score_as_{color}"] = points / len(subset) if subset else None
    return result


def should_promote(evaluation, threshold):
    # Do not improve an apparent win rate by silently excluding unfinished games.
    return (evaluation["completed"] == evaluation["games"] and evaluation["games"] > 0
            and evaluation["score"] is not None and evaluation["score"] >= threshold)


def evaluate_models(candidate, opponent, config, games, opening_moves, seed, device, batch_size):
    envs, candidate_colors, colors = paired_openings(config, games, opening_moves, seed)
    searches = [MCTS(candidate, config.c_puct, config.num_simulations),
                MCTS(opponent, config.c_puct, config.num_simulations)]
    while True:
        active = [i for i, env in enumerate(envs) if not env.game_over and len(env.moves_history) < config.max_moves]
        if not active:
            break
        # Group from a snapshot of this ply: an environment is moved exactly once.
        groups = [[i for i in active if (colors[i] == candidate_colors[i]) == (side == 0)]
                  for side in range(2)]
        for search, group in zip(searches, groups):
            for start in range(0, len(group), batch_size):
                ids = group[start:start + batch_size]
                policies = search.get_action_probs_batch(
                    [envs[i] for i in ids], [colors[i] for i in ids], device=device,
                    temperature=1.0, add_root_noise=False,
                    min_moves_before_pass=config.min_moves_before_pass,
                )
                for i, policy in zip(ids, policies):
                    apply_action(envs[i], colors[i], index_to_action(int(np.argmax(policy)), config.board_size))
                    colors[i] = other_color(colors[i])
    records = [dict(outcome_record(env), candidate_color=color)
               for env, color in zip(envs, candidate_colors)]
    return summarize_evaluation(records), records

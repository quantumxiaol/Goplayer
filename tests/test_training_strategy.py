"""Strategy tests use canned positions/policies only; never call Trainer.run/train_step."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Goplayer.goenv import GoEnv
from rl.augmentation import transform_sample
from rl.mcts import MCTS, temperature_policy
from rl.replay_buffer import ReplayBuffer
from rl.selfplay import (SearchConfig, finish_samples, outcome_record, paired_openings,
                         play_self_play_batch, should_promote, summarize_evaluation, evaluate_models)
from scripts.train import parse_args, summarize_games


def config(**overrides):
    defaults = dict(board_size=9, komi=5.5, num_simulations=2, c_puct=1.5,
                    min_moves_before_pass=0, max_moves=4, temperature_moves=0,
                    final_temperature=0, dirichlet_alpha=0.1, noise_fraction=0.25)
    return SearchConfig(**(defaults | overrides))


class CountingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, states):
        self.batches.append(len(states))
        size = states.shape[-1]
        return torch.zeros((len(states), size * size + 1)), torch.zeros((len(states), 1))


class SearchStrategyTests(unittest.TestCase):
    def test_sampling_temperature_does_not_mutate_training_target(self):
        visits = np.array([0.6, 0.3, 0.1], dtype=np.float32)
        saved = visits.copy()
        sampled = temperature_policy(visits, 0)
        np.testing.assert_array_equal(sampled, [1, 0, 0])
        np.testing.assert_array_equal(visits, saved)

    def test_root_noise_is_legal_normalized_reproducible_and_opt_in(self):
        env = GoEnv(9)
        env.place_stone(0, 0, 'black')
        search = MCTS(CountingNet(), num_simulations=1)
        baseline = search.get_action_probs(env, 'white', allow_pass=False)
        noisy = search.get_action_probs(env, 'white', allow_pass=False, add_root_noise=True,
                                       rng=np.random.default_rng(123))
        again = search.get_action_probs(env, 'white', allow_pass=False, add_root_noise=True,
                                       rng=np.random.default_rng(123))
        np.testing.assert_array_equal(noisy, again)
        self.assertFalse(np.allclose(baseline, noisy))
        self.assertEqual(noisy[0], 0)
        self.assertEqual(noisy[-1], 0)
        self.assertAlmostEqual(float(noisy.sum()), 1, places=6)
        np.testing.assert_array_equal(search.get_action_probs(env, 'white', allow_pass=False), baseline)

    def test_noise_is_added_only_at_root(self):
        search = MCTS(CountingNet(), num_simulations=4)
        with patch.object(search, '_add_root_noise', wraps=search._add_root_noise) as add_noise:
            search.get_action_probs(GoEnv(9), 'black', add_root_noise=True)
        self.assertEqual(add_noise.call_count, 1)

    def test_batched_search_matches_independent_search_without_noise(self):
        envs = [GoEnv(9), GoEnv(9)]
        envs[1].place_stone(0, 0, 'black')
        net = CountingNet()
        batched = MCTS(net, num_simulations=3).get_action_probs_batch(envs, ['black', 'white'])
        self.assertEqual(net.batches, [2, 2, 2])
        for env, color, result in zip(envs, ['black', 'white'], batched):
            single = MCTS(CountingNet(), num_simulations=3).get_action_probs(env, color)
            np.testing.assert_array_equal(result, single)

    def test_minimum_pass_ply_is_respected(self):
        search = MCTS(CountingNet(), num_simulations=1)
        env = GoEnv(9)
        self.assertEqual(search.get_action_probs(env, 'black', min_moves_before_pass=2)[-1], 0)
        env.place_stone(2, 2, 'black')
        env.place_stone(6, 6, 'white')
        self.assertGreater(search.get_action_probs(env, 'black', min_moves_before_pass=2)[-1], 0)

    def test_legal_terminal_draw_has_zero_search_value(self):
        env = GoEnv(13)
        moves = [(0, 2), (0, 0), (1, 1), (0, 1), (1, 0), (0, 0), None, (8, 8), (8, 7), None, None]
        for i, move in enumerate(moves):
            color = 'black' if i % 2 == 0 else 'white'
            if move is None:
                env.register_pass(color)
            else:
                self.assertTrue(env.place_stone(*move, color))
        self.assertEqual(env.calculate_area_score(), (4, 4))
        self.assertEqual(env.judge_winner()['winner'], 'draw')
        search = MCTS(CountingNet(), num_simulations=1)
        self.assertEqual(search._terminal_value(env, 'black'), 0)
        self.assertEqual(search._terminal_value(env, 'white'), 0)

    def test_clone_retains_ko_pass_komi_and_independence(self):
        env = GoEnv(9, komi=2)
        env.place_stone(0, 0, 'black')
        env.register_pass('white')
        before = copy.deepcopy(env.__dict__)
        clone = env.clone_for_search()
        self.assertEqual(clone.position_history, env.position_history)
        self.assertEqual(clone.consecutive_passes, 1)
        self.assertEqual(clone.komi, 2)
        self.assertEqual(clone.undo_stack, [])
        clone.place_stone(2, 2, 'black')
        clone.register_pass('white')
        self.assertEqual(clone.undo_stack, [])
        self.assertEqual(env.__dict__, before)

    def test_all_symmetries_keep_stone_and_policy_aligned_and_pass_unchanged(self):
        state = torch.zeros(3, 9, 9)
        state[0, 1, 3] = 1
        state[2] = 1
        policy = np.zeros(82, dtype=np.float32)
        policy[12], policy[-1] = 0.75, 0.25
        for symmetry in range(8):
            transformed, probs = transform_sample(state, policy, symmetry)
            self.assertEqual(int(transformed[0].flatten().argmax()), int(probs[:-1].argmax()))
            self.assertEqual(probs[-1], 0.25)
            self.assertTrue(torch.equal(transformed[2], state[2]))
            self.assertAlmostEqual(float(probs.sum()), 1)


class DatasetAndArenaTests(unittest.TestCase):
    def test_replay_sampling_accepts_deque_without_training(self):
        buffer = ReplayBuffer(capacity=3)
        for i in range(4):
            buffer.add(i, [i], float(i))
        states, policies, values = buffer.sample(10)
        self.assertEqual(set(states), {1, 2, 3})
        self.assertEqual(len(states), 3)
        for state, policy, value in zip(states, policies, values):
            self.assertEqual(policy, [state])
            self.assertEqual(value, float(state))
        self.assertEqual(ReplayBuffer().sample(1), ([], [], []))

    def test_draw_labels_zero_and_truncation_has_no_samples(self):
        history = [(torch.zeros(3, 9, 9), np.ones(82)/82, color) for color in ['black', 'white']]
        self.assertEqual([s[2] for s in finish_samples(history, {'termination': 'double_pass', 'winner': 'draw'})], [0, 0])
        self.assertEqual(finish_samples(history, {'termination': 'max_moves', 'winner': None}), [])
        self.assertEqual([s[2] for s in finish_samples(history, {'termination': 'double_pass', 'winner': 'black'})], [1, -1])

    def test_capped_position_is_not_adjudicated_as_a_win(self):
        env = GoEnv(9)
        record = outcome_record(env)
        self.assertEqual(record['termination'], 'max_moves')
        self.assertIsNone(record['winner'])
        self.assertFalse(env.game_over)

    def test_soft_target_survives_deterministic_sampling_in_collection(self):
        # Two canned passes, no model inference, no optimizer, no actual training.
        def scripted_search(search, envs, colors, **kwargs):
            self.assertEqual(kwargs['temperature'], 1)
            self.assertTrue(kwargs['add_root_noise'])
            search.last_root_values = [0] * len(envs)
            policy = np.zeros(82, dtype=np.float32)
            policy[0], policy[-1] = 0.4, 0.6
            return [policy.copy() for _ in envs]
        with patch.object(MCTS, 'get_action_probs_batch', scripted_search):
            results = play_self_play_batch(None, config(), 2, 'cpu', np.random.default_rng(0))
        for samples, record in results:
            self.assertEqual(record['termination'], 'double_pass')
            self.assertEqual(len(samples), 2)
            self.assertAlmostEqual(float(samples[0][1][0]), 0.4)
            self.assertAlmostEqual(float(samples[0][1][-1]), 0.6)

    def test_paired_openings_share_board_but_not_mutable_state(self):
        envs, colors, to_play = paired_openings(config(max_moves=20), 4, 4, 12)
        self.assertEqual(colors, ['black', 'white', 'black', 'white'])
        self.assertEqual(to_play, ['black'] * 4)
        self.assertEqual(envs[0].grid, envs[1].grid)
        repeat, _, _ = paired_openings(config(max_moves=20), 4, 4, 12)
        self.assertEqual(envs[2].grid, repeat[2].grid)
        envs[0].grid[0][0] = 'black'
        self.assertIsNot(envs[0].grid, envs[1].grid)

    def test_arena_uses_no_noise_and_draws_are_half_points(self):
        def passes(search, envs, colors, **kwargs):
            self.assertFalse(kwargs['add_root_noise'])
            policy = np.zeros(82, dtype=np.float32)
            policy[-1] = 1
            return [policy.copy() for _ in envs]
        with patch.object(MCTS, 'get_action_probs_batch', passes):
            summary, records = evaluate_models(None, None, config(komi=0), 2, 0, 1, 'cpu', 2)
        self.assertEqual(summary['draws'], 2)
        self.assertEqual(summary['score'], 0.5)
        self.assertEqual([r['moves'] for r in records], [2, 2])
        self.assertFalse(should_promote(summary, 0.55))

    def test_truncated_arena_cannot_promote_or_count_as_draw(self):
        summary = summarize_evaluation([
            {'winner': 'black', 'candidate_color': 'black'},
            {'winner': None, 'candidate_color': 'white'},
        ])
        self.assertEqual(summary['wins'], 1)
        self.assertEqual(summary['draws'], 0)
        self.assertEqual(summary['truncated'], 1)
        self.assertFalse(should_promote(summary, 0.55))
        summary = summarize_evaluation([
            {'winner': 'black', 'candidate_color': 'black'},
            {'winner': 'white', 'candidate_color': 'white'},
        ])
        self.assertTrue(should_promote(summary, 0.55))

    def test_summary_excludes_truncation_from_win_rate(self):
        env = GoEnv(9)
        records = []
        for winner, termination in [('black', 'double_pass'), ('draw', 'double_pass'), (None, 'max_moves')]:
            record = outcome_record(env) | dict(winner=winner, termination=termination, policy_entropy=1,
                                               policy_max=0.5, root_value_black=None, root_value_white=None)
            records.append(record)
        metrics = summarize_games(records)
        self.assertEqual(metrics['black_score_rate'], 0.75)
        self.assertEqual(metrics['draws'], 1)
        self.assertEqual(metrics['truncated'], 1)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.dotenv = patch('scripts.train.load_dotenv')
        self.dotenv.start()

    def tearDown(self):
        self.environment.stop()
        self.dotenv.stop()

    def test_13_preset_and_cli_override(self):
        args = parse_args(['--config', 'configs/train-13x13.json', '--iterations', '7', '--device', 'cpu'])
        self.assertEqual(args.board_size, 13)
        self.assertEqual(args.num_simulations, 320)
        self.assertEqual(args.min_moves_before_pass, 100)
        self.assertEqual(args.iterations, 7)
        self.assertEqual(args.device, 'cpu')

    def test_board_defaults_scale_without_config(self):
        args = parse_args(['--board-size', '13'])
        self.assertEqual(args.num_simulations, 320)
        self.assertEqual(args.temperature_moves, 80)

    def test_invalid_configuration_rejected(self):
        invalid = [['--eval-games', '3'], ['--num-simulations', '0'], ['--noise-fraction', '2'],
                   ['--max-moves', '10'], ['--buffer-size', '1'], ['--run-name', '../old'],
                   ['--promotion-threshold', '0.5'], ['--final-temperature', '-1'], ['--seed', '-1']]
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_unknown_config_key_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'num_simulatoins': 320}))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--config', str(path)])

    def test_invalid_json_types_are_rejected(self):
        for defaults in [{'board_size': None}, {'batch_size': True}, {'augment': 'false'}, {'run_name': []}]:
            with self.subTest(defaults=defaults), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(defaults))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(['--config', str(path)])


if __name__ == '__main__':
    unittest.main()

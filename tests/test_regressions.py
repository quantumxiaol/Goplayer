import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from Goplayer.goenv import GoEnv
from Goplayer.goplayer import AIPlayer, HumanPlayer
from rl.mcts import MCTS


class PassHistoryTests(unittest.TestCase):
    def test_pass_undo_preserves_stones_and_turn(self):
        env = GoEnv(9)
        env.place_stone(0, 0, 'black')
        positions = env.position_history.copy()
        env.register_pass('white')
        self.assertEqual(env.undo(), (-1, -1, 'white'))
        self.assertEqual(env.grid[0][0], 'black')
        self.assertEqual(env.position_history, positions)
        self.assertEqual(env.consecutive_passes, 0)
        self.assertEqual(env.undo(), (0, 0, 'black'))

    def test_empty_board_double_pass_can_be_undone(self):
        env = GoEnv(9)
        env.register_pass()
        env.register_pass()
        self.assertTrue(env.game_over)
        self.assertEqual(env.undo(), (-1, -1, 'white'))
        self.assertFalse(env.game_over)
        self.assertEqual(env.consecutive_passes, 1)
        self.assertEqual(env.undo(), (-1, -1, 'black'))
        self.assertEqual(env.consecutive_passes, 0)
        self.assertEqual(env.moves_history, [])

    def test_stone_after_pass_restores_pass_on_undo(self):
        env = GoEnv(9)
        env.register_pass('black')
        env.place_stone(2, 3, 'white')
        self.assertEqual(env.consecutive_passes, 0)
        env.undo()
        self.assertEqual(env.consecutive_passes, 1)
        self.assertEqual(env.moves_history, [(-1, -1, 'black')])


class APICoordinateTests(unittest.TestCase):
    def player(self, *replies):
        player = AIPlayer.__new__(AIPlayer)
        player.color, player.model, player.tourance = 'black', 'mock', 3
        create = Mock(side_effect=[
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
            for reply in replies
        ])
        player.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return player

    def test_corners_and_pass(self):
        for text, expected in [('1,1', (0, 0)), ('9,9', (8, 8)), ('1, 9', (0, 8)), ('-1,-1', (-1, -1))]:
            with self.subTest(text=text):
                self.assertEqual(self.player(text).get_move_from_gpt(GoEnv(9)), expected)

    def test_retry_uses_same_coordinate_conversion(self):
        env = GoEnv(9)
        env.place_stone(0, 0, 'white')
        player = self.player('1,1', '9,9')
        self.assertEqual(player.get_move_from_gpt(env), (8, 8))
        self.assertEqual(player.client.chat.completions.create.call_count, 2)
        self.assertIn('black', player.client.chat.completions.create.call_args_list[0].kwargs['messages'][1]['content'])

    def test_invalid_coordinates_retry(self):
        player = self.player('0,0', '10,9', '1,9')
        self.assertEqual(player.get_move_from_gpt(GoEnv(9)), (0, 8))

    def test_cancel_prevents_request_and_retry(self):
        player = self.player('0,0', '1,1')
        self.assertIsNone(player.get_move_from_gpt(GoEnv(9), should_cancel=lambda: True))
        player.client.chat.completions.create.assert_not_called()
        self.assertIsNone(player.get_move_from_gpt(
            GoEnv(9), should_cancel=lambda: player.client.chat.completions.create.call_count > 0,
        ))
        self.assertEqual(player.client.chat.completions.create.call_count, 1)

    def test_human_click_uses_row_y_and_column_x(self):
        board = SimpleNamespace(margin=25, cell_size=50, place_stone=Mock(return_value=True))
        event = SimpleNamespace(pos=lambda: SimpleNamespace(x=lambda: 425, y=lambda: 25))
        HumanPlayer('black').make_move(board, event)
        board.place_stone.assert_called_once_with(0, 8, 'black')


class UniformNet(torch.nn.Module):
    def forward(self, state):
        size = state.shape[-1]
        return torch.zeros((1, size * size + 1)), torch.zeros((1, 1))


class MCTSBoundaryTests(unittest.TestCase):
    def test_nonpositive_simulations_rejected(self):
        for count in [0, -1]:
            with self.assertRaises(ValueError):
                MCTS(UniformNet(), num_simulations=count)

    def test_low_simulations_only_assign_legal_moves(self):
        env = GoEnv(9)
        env.place_stone(0, 1, 'black')
        env.place_stone(1, 0, 'black')
        legal = {r * 9 + c for r, c in env.legal_moves('white')}
        for count in [1, 2]:
            for temperature in [0, 1e-3, 1]:
                with self.subTest(count=count, temperature=temperature):
                    probs = MCTS(UniformNet(), num_simulations=count).get_action_probs(
                        env, 'white', temperature=temperature, allow_pass=False,
                    )
                    self.assertTrue(np.isfinite(probs).all())
                    self.assertAlmostEqual(float(probs.sum()), 1, places=6)
                    self.assertTrue(set(np.flatnonzero(probs)).issubset(legal))
                    self.assertEqual(probs[0], 0)  # suicide
                    self.assertEqual(probs[1], 0)  # occupied
                    self.assertEqual(probs[-1], 0)  # forbidden Pass

    def test_no_legal_stones_still_allows_pass(self):
        env = GoEnv(1)  # The sole empty point is suicide.
        for count in [1, 2]:
            probs = MCTS(UniformNet(), num_simulations=count).get_action_probs(env, 'black', allow_pass=False)
            np.testing.assert_array_equal(probs, [0, 1])

    def test_search_rejects_finished_game(self):
        env = GoEnv(9)
        env.register_pass()
        env.register_pass()
        with self.assertRaises(ValueError):
            MCTS(UniformNet(), num_simulations=1).get_action_probs(env, 'black')


if __name__ == '__main__':
    unittest.main()

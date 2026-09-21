"""Persistence/inference checks only: no optimizer updates or real training runs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Goplayer.goenv import GoEnv
from rl.augmentation import transform_sample
from rl.checkpoints import model_from_checkpoint, unpack_checkpoint
from rl.data_archive import GameArchive, game_sgf, load_archive, read_samples, split_for_game
from rl.encoder import encode_state
from rl.learning import LOSS_KEYS, loss_terms, validate_model
from rl.mcts import MCTS
from rl.net import GoNet
from rl.replay_buffer import ReplayBuffer
from rl.selfplay import outcome_record, SearchConfig
from rl.utils import save_checkpoint
from scripts.train import load_model_weights


def fixture(features='pass-v2'):
    env = GoEnv(9)
    history = []
    for color, move in [('black', (1, 2)), ('white', (3, 4)), ('black', (-1, -1)), ('white', (-1, -1))]:
        state = encode_state(env, color, features)
        policy = np.zeros(82, dtype=np.float32)
        policy[-1 if move == (-1, -1) else move[0] * 9 + move[1]] = 1
        history.append((state, policy, -1.0 if color == "black" else 1.0))
        if move == (-1, -1):
            env.register_pass(color)
        else:
            assert env.place_stone(*move, color)
    return history, outcome_record(env)


class ArchiveTests(unittest.TestCase):
    def test_roundtrip_samples_boards_moves_sgf_and_disjoint_split(self):
        samples, record = fixture()
        with tempfile.TemporaryDirectory() as directory:
            archive = GameArchive(Path(directory) / 'data', 9, 5.5, 'pass-v2', 42, 0.5)
            splits = {}
            for game in range(1, 9):
                split, game_id = archive.save_game(1, game, samples, record)
                splits[game_id] = split
            self.assertEqual(set(splits.values()), {'train', 'validation'})
            path = archive.root / next(iter(splits))
            loaded = read_samples(path / 'samples.npz', 9, 'pass-v2')
            for original, restored in zip(samples, loaded):
                torch.testing.assert_close(original[0], restored[0])
                np.testing.assert_array_equal(original[1], restored[1])
                self.assertEqual(original[2], restored[2])
            stored = json.loads((path / 'game.json').read_text())
            env = GoEnv(stored['board_size'], stored['komi'])
            for row, col, color in stored['move_sequence']:
                if row == -1:
                    env.register_pass(color)
                else:
                    self.assertTrue(env.place_stone(row, col, color))
            self.assertEqual(env.grid, stored['final_board'])
            self.assertEqual(env.calculate_area_score(), (stored['black_score'], stored['white_score']))
            self.assertIn(';B[cb];W[ed];B[];W[]', (path / 'game.sgf').read_text())
            train, val = ReplayBuffer(100), ReplayBuffer(100)
            index = load_archive(archive.root, train, val, 9, 5.5, 'pass-v2')
            self.assertEqual(len(train), 4 * list(splits.values()).count('train'))
            self.assertEqual(len(val), 4 * list(splits.values()).count('validation'))
            self.assertEqual({r['game_id']: r['split'] for r in index}, splits)
            with self.assertRaises(FileExistsError):
                archive.save_game(1, 1, samples, record)
            with self.assertRaises(ValueError):
                load_archive(archive.root, train, val, 9, 5.5, 'stones-v1')

    def test_truncation_is_inspectable_but_has_no_training_samples(self):
        env = GoEnv(9)
        env.place_stone(0, 0, 'black')
        record = outcome_record(env)
        with tempfile.TemporaryDirectory() as directory:
            archive = GameArchive(Path(directory) / 'data', 9, 5.5, 'stones-v1', 0, 0.1)
            _, game_id = archive.save_game(1, 1, [], record)
            self.assertEqual(read_samples(archive.root / game_id / 'samples.npz', 9, 'stones-v1'), [])
            self.assertNotIn('RE[', game_sgf(record))
            self.assertIn('max_moves', game_sgf(record))
            (archive.root / '.pending-interrupted').mkdir()
            index = load_archive(archive.root, ReplayBuffer(), ReplayBuffer(), 9, 5.5, 'stones-v1')
            self.assertEqual(len(index), 1)

    def test_trainer_archives_and_logs_with_collection_and_updates_mocked(self):
        from scripts.train import Trainer, parse_args
        import csv
        samples, record = fixture()
        record.update(policy_entropy=0.0, policy_max=1.0, root_value_black=0.0, root_value_white=0.0)
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--board-size', '9', '--input-features', 'pass-v2', '--device', 'cpu',
                               '--checkpoint-dir', directory + '/checkpoints', '--log-dir', directory + '/logs',
                               '--run-name', 'mocked', '--iterations', '1', '--games-per-iteration', '8',
                               '--batch-size', '2', '--channels', '4', '--res-blocks', '1',
                               '--validation-fraction', '0.5', '--no-tensorboard'])
            with patch('scripts.train.play_self_play_batch', return_value=[(samples, record)] * 8), \
                 patch('scripts.train.update_model', return_value=dict.fromkeys(LOSS_KEYS, 0.0)) as update, \
                 patch('torch.optim.Adam.step', side_effect=AssertionError('No training allowed')), \
                 contextlib.redirect_stdout(io.StringIO()):
                trainer = Trainer(args)
                trainer.run()
            self.assertEqual(update.call_count, args.train_steps_per_iteration)
            with (trainer.log_dir / 'train_metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertGreater(int(rows[0]['validation_samples']), 0)
            self.assertEqual(len(trainer.buffer) + len(trainer.validation_buffer), 32)
            self.assertEqual(len(list(trainer.archive.root.glob('iter-*/samples.npz'))), 8)
            self.assertTrue((trainer.checkpoint_dir / 'initial_model.pth').exists())
            restored_train, restored_val = ReplayBuffer(), ReplayBuffer()
            load_archive(trainer.archive.root, restored_train, restored_val, 9, 5.5, 'pass-v2')
            self.assertEqual(len(restored_train), len(trainer.buffer))
            self.assertEqual(len(restored_val), len(trainer.validation_buffer))

    def test_split_reproducible_and_independent_of_position(self):
        first = [split_for_game(str(i), 42, 0.1) for i in range(100)]
        self.assertEqual(first, [split_for_game(str(i), 42, 0.1) for i in range(100)])
        self.assertIn('validation', first)


class FeatureAndCheckpointTests(unittest.TestCase):
    def test_pass_plane_resets_on_stone_and_undo_and_is_symmetry_invariant(self):
        env = GoEnv(9)
        before = encode_state(env, 'white', 'pass-v2')
        env.register_pass('black')
        after = encode_state(env, 'white', 'pass-v2')
        torch.testing.assert_close(before[:3], after[:3])
        self.assertEqual(float(after[3].sum()), 81)
        self.assertEqual(float(before[3].sum()), 0)
        policy = np.ones(82, dtype=np.float32) / 82
        for symmetry in range(8):
            transformed, _ = transform_sample(after, policy, symmetry)
            torch.testing.assert_close(transformed[3], after[3])
        env.place_stone(2, 2, 'white')
        self.assertEqual(float(encode_state(env, 'black', 'pass-v2')[3].sum()), 0)
        env.undo()
        self.assertEqual(float(encode_state(env, 'white', 'pass-v2')[3].sum()), 81)
        env.undo()
        self.assertEqual(float(encode_state(env, 'white', 'pass-v2')[3].sum()), 0)
        self.assertEqual(encode_state(env, 'white').shape[0], 3)

    def test_old_raw_and_new_checkpoints_load_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for features in ('stones-v1', 'pass-v2'):
                model = GoNet(9, 4, 1, features)
                path = Path(directory) / f'{features}.pth'
                if features == 'stones-v1':
                    torch.save(model.state_dict(), path)
                else:
                    save_checkpoint(path, model, metadata=dict(input_features=features, komi=5.5))
                loaded, _, spec = model_from_checkpoint(path, board_size=9, komi=5.5)
                self.assertEqual(spec['input_features'], features)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value)
                other = GoNet(9, 4, 1, 'pass-v2' if features == 'stones-v1' else 'stones-v1')
                with self.assertRaisesRegex(ValueError, 'Input version mismatch'):
                    load_model_weights(other, path, 9, 5.5)
                with self.assertRaises(ValueError):
                    model_from_checkpoint(path, board_size=13)
            with self.assertRaises(ValueError):
                model_from_checkpoint(path, komi=2)
            with self.assertRaises(ValueError):
                unpack_checkpoint(dict(model_state_dict=model.state_dict(), metadata=dict(input_features='stones-v1')))

    def test_mcts_uses_model_input_version(self):
        class PassNet(torch.nn.Module):
            input_features = 'pass-v2'
            def forward(self, states):
                assert states.shape[1] == 4 and torch.all(states[:, 3] == 1)
                return torch.zeros(len(states), 82), torch.zeros(len(states), 1)
        env = GoEnv(9)
        env.register_pass('black')
        probs = MCTS(PassNet(), num_simulations=1).get_action_probs(env, 'white')
        self.assertAlmostEqual(float(probs.sum()), 1)

    def test_export_both_input_versions_has_correct_shapes_and_outputs(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from scripts.export_onnx import main
        with tempfile.TemporaryDirectory() as directory:
            for features in ('stones-v1', 'pass-v2'):
                model = GoNet(9, 4, 1, features).eval()
                path = Path(directory) / f'{features}.pth'
                output = path.with_suffix('.onnx')
                save_checkpoint(path, model, metadata=dict(input_features=features, komi=5.5))
                argv = ['export_onnx.py', '--checkpoint', str(path), '--output', str(output)]
                with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
                graph = onnx.load(output)
                self.assertEqual(graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value, model.input_channels)
                env = GoEnv(9)
                env.register_pass('black')
                state = encode_state(env, 'white', features)[None]
                with torch.inference_mode():
                    expected = model(state)
                actual = ReferenceEvaluator(graph).run(None, {'board_state': state.numpy()})
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a, e.numpy(), atol=1e-5)
                self.assertEqual(json.loads(output.with_suffix('.json').read_text())['input_features'], features)


class DiagnosticsTests(unittest.TestCase):
    def test_kl_equals_cross_entropy_minus_same_batch_entropy(self):
        target = torch.tensor([[0.75, 0.25], [0.0, 1.0]])
        logits = target.clamp_min(1e-30).log()
        terms = loss_terms(logits, torch.tensor([0., 1.]), target, torch.tensor([0., 1.]))
        self.assertAlmostEqual(float(terms['policy_kl']), 0, places=6)
        self.assertAlmostEqual(float(terms['policy_loss']), float(terms['target_entropy']), places=6)
        self.assertEqual(float(terms['value_loss']), 0)
        uniform = loss_terms(torch.zeros_like(logits), torch.zeros(2), target, torch.zeros(2))
        self.assertGreater(float(uniform['policy_kl']), 0)
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))

    def test_validation_does_not_update_weights_or_batchnorm_and_weights_partial_batch(self):
        model = GoNet(9, 4, 1, 'pass-v2').train()
        samples, _ = fixture()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        result = validate_model(model, samples, 'cpu', 3)
        whole = validate_model(model, samples, 'cpu', 4)
        self.assertTrue(model.training)
        self.assertEqual(result['validation_samples'], 4)
        for key in LOSS_KEYS:
            self.assertAlmostEqual(result[f'val_{key}'], whole[f'val_{key}'], places=5)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertIsNone(validate_model(model, [], 'cpu')['val_policy_kl'])

    def test_comparison_orchestration_with_mocked_updates_never_trains(self):
        from scripts.compare_updates import parse_args, run_comparison
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = GameArchive(root / 'data', 9, 5.5, 'pass-v2', 42, 0.5)
            samples, record = fixture()
            for game in range(1, 9):
                archive.save_game(1, game, samples, record)
            model = GoNet(9, 4, 1, 'pass-v2')
            config = SearchConfig(9, 5.5, 2, 1.5, 0, 10, 0, 0, 0.1, 0.25)
            path = root / 'initial.pth'
            save_checkpoint(path, model, metadata=dict(komi=5.5, search_config=config.__dict__))
            args = parse_args(['--data-dir', str(archive.root), '--checkpoint', str(path),
                               '--output-dir', str(root / 'comparison'), '--updates', '2', '4',
                               '--rounds', '2', '--batch-size', '2', '--skip-arena', '--device', 'cpu'])
            seen = []
            def fake_update(candidate, optimizer, batch, device, **kwargs):
                # All calls must still see the unchanged initial weights.
                for key, value in candidate.state_dict().items():
                    torch.testing.assert_close(value, model.state_dict()[key])
                seen.append([(s.clone(), p.copy(), v) for s, p, v in batch])
                return dict.fromkeys(LOSS_KEYS, 0.0)
            with patch('scripts.compare_updates.update_model', side_effect=fake_update) as update, \
                 patch('torch.optim.Adam.step', side_effect=AssertionError('No training allowed')), \
                 contextlib.redirect_stdout(io.StringIO()):
                results = run_comparison(args)
            self.assertEqual(update.call_count, 12)
            self.assertEqual([r['updates'] for r in results], [4, 8])
            for small, large in zip(seen[:4], seen[4:8]):
                for a, b in zip(small, large):
                    torch.testing.assert_close(a[0], b[0])
                    np.testing.assert_array_equal(a[1], b[1])
            self.assertTrue((root / 'comparison' / 'comparison.json').exists())


if __name__ == '__main__':
    unittest.main()

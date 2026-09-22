"""Exercise CLI orchestration using canned games; no search or training is run."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Goplayer.goenv import GoEnv
from rl.selfplay import SearchConfig, outcome_record, summarize_evaluation
from scripts.evaluate_checkpoints import parse_args, run_evaluation


class CheckpointEvaluationTests(unittest.TestCase):
    def test_multiple_seeds_share_search_budget_and_keep_candidate_perspective(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, opponent = root / 'candidate.pth', root / 'opponent.pth'
            candidate.write_bytes(b'candidate fixture')
            opponent.write_bytes(b'opponent fixture')
            args = parse_args(['--candidate', str(candidate), '--opponent', str(opponent),
                               '--output-dir', str(root / 'results'), '--seeds', '101', '102',
                               '--games-per-seed', '2', '--num-simulations', '123', '--device', 'cpu'])
            config = SearchConfig(9, 5.5, 320, 1.5, 0, 50, 10, 0.25, 0.1, 0.25)
            spec = dict(board_size=9, input_features='stones-v1')
            metadata = dict(komi=5.5, search_config=config.__dict__)
            calls = []
            def canned(a, b, shared, games, opening, seed, device, batch_size):
                calls.append((a, b, shared.num_simulations, games, opening, seed))
                env = GoEnv(9)
                env.register_pass('black')
                env.register_pass('white')
                records = [dict(outcome_record(env), candidate_color=color) for color in ['black', 'white']]
                return summarize_evaluation(records), records
            with patch('scripts.evaluate_checkpoints.model_from_checkpoint',
                       side_effect=[('candidate', metadata, spec), ('opponent', metadata, spec)]), \
                 patch('scripts.evaluate_checkpoints.evaluate_models', side_effect=canned), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = run_evaluation(args)
            self.assertEqual(calls, [('candidate', 'opponent', 123, 2, 8, 101),
                                     ('candidate', 'opponent', 123, 2, 8, 102)])
            self.assertEqual(result['overall']['games'], 4)
            self.assertEqual(result['overall']['score'], 0.5)
            self.assertEqual(result['overall']['score_as_black'], 0)
            self.assertEqual(result['overall']['score_as_white'], 1)
            self.assertEqual(len(list(args.output_dir.glob('*.sgf'))), 4)
            self.assertEqual(len((args.output_dir / 'games.jsonl').read_text().splitlines()), 4)
            self.assertEqual(json.loads((args.output_dir / 'results.json').read_text()), result)
            with self.assertRaises(FileExistsError):
                run_evaluation(args)

    def test_invalid_cli_settings_rejected(self):
        required = ['--candidate', 'a.pth', '--opponent', 'b.pth', '--output-dir', 'out']
        for extra in [['--games-per-seed', '3'], ['--seeds', '1', '1'], ['--seeds', '-1'],
                      ['--opening-moves', '0'], ['--num-simulations', '0'], ['--batch-size', '0']]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(required + extra)


if __name__ == '__main__':
    unittest.main()

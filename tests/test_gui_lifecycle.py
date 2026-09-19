import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import threading
import time
import unittest
from unittest.mock import patch

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication

from Goplayer.goboard import AIMoveWorker, GoBoard
from Goplayer.goplayer import AIPlayer


class DelayedAI(AIPlayer):
    def __init__(self, move=(3, 4)):
        self.color = 'white'
        self.move = move
        self.started = threading.Event()
        self.released = threading.Event()

    def get_move_from_gpt(self, board, should_cancel=None):
        self.started.set()
        if not self.released.wait(5):
            raise TimeoutError('Test did not release mock request')
        return self.move


class GUILifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.boards = []
        self.players = []
        self.dialogs = patch('Goplayer.goboard.QMessageBox.information')
        self.dialogs.start()

    def flush(self):
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def tearDown(self):
        for player in self.players:
            player.released.set()
        AIMoveWorker.shutdown()
        self.flush()
        for board in self.boards:
            if not sip.isdeleted(board):
                board._stop_ai_worker()
                sip.delete(board)
        self.flush()
        self.dialogs.stop()
        self.assertEqual(AIMoveWorker.active_workers, set())

    def board(self):
        board = GoBoard(size=9)
        self.boards.append(board)
        return board

    def start_request(self, board, move=(3, 4)):
        player = DelayedAI(move)
        self.players.append(player)
        board.players['white'] = player
        board.current_player = player
        board._start_ai_move_async()
        self.assertTrue(player.started.wait(2))
        return player, board.ai_thread

    def test_reset_keeps_blocked_worker_alive_and_ignores_result(self):
        board = self.board()
        player, worker = self.start_request(board)
        started = time.monotonic()
        board.reset()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(worker.isRunning())
        self.assertFalse(sip.isdeleted(worker))
        player.released.set()
        self.assertTrue(worker.wait(2000))
        self.flush()
        self.assertEqual(board.moves_history, [])
        self.assertFalse(board.ai_thinking)

    def test_queued_old_result_and_cleanup_do_not_affect_new_request(self):
        board = self.board()
        old_player, old_worker = self.start_request(board, (1, 1))
        old_player.released.set()
        self.assertTrue(old_worker.wait(2000))  # Result is queued, not yet delivered.
        board.reset()
        new_player, new_worker = self.start_request(board, (2, 3))
        self.flush()
        self.assertIs(board.ai_thread, new_worker)
        self.assertTrue(board.ai_thinking)
        self.assertEqual(board.moves_history, [])
        new_player.released.set()
        self.assertTrue(new_worker.wait(2000))
        self.flush()
        self.assertEqual(board.moves_history, [(2, 3, 'white')])
        self.assertFalse(board.ai_thinking)

    def test_board_deletion_does_not_destroy_running_worker(self):
        board = self.board()
        player, worker = self.start_request(board)
        sip.delete(board)
        self.assertTrue(worker.isRunning())
        self.assertFalse(sip.isdeleted(worker))
        player.released.set()
        self.assertTrue(worker.wait(2000))
        self.flush()

    def test_pass_undo_restores_correct_player_and_board(self):
        board = self.board()
        board.place_stone(0, 0, 'black')
        board.switch_player()
        board.pass_turn()
        board.pass_turn()
        self.assertTrue(board.game_over)
        board.undo_move()
        self.assertFalse(board.game_over)
        self.assertEqual(board.current_player.color, 'black')
        self.assertEqual(board.consecutive_passes, 1)
        self.assertEqual(board.grid[0][0], 'black')
        board.undo_move()
        self.assertEqual(board.current_player.color, 'white')
        self.assertEqual(board.consecutive_passes, 0)

    def test_api_pass_is_recorded_instead_of_random_stone(self):
        board = self.board()
        player, worker = self.start_request(board, (-1, -1))
        player.released.set()
        self.assertTrue(worker.wait(2000))
        self.flush()
        self.assertEqual(board.moves_history, [(-1, -1, 'white')])
        self.assertEqual(board.current_player.color, 'black')


if __name__ == '__main__':
    unittest.main()

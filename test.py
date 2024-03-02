from stat_engine import NewStatEngine as Engine
from cli_game import Board

N = 3
X = 1
board = Board(4, N, X)

board.charges = [2, 3]
board._active_items = {'handcuffs': 0.5, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0}
board.current_turn = 0

# Player
board.items[0] = {'handcuffs': 2, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0}

# Opponent
board.items[1] = {'handcuffs': 0, 'magnifying_glass': 3, 'beer': 0, 'cigarettes': 1, 'saw': 1}

board.chamber_public = False
board._skip_next = False

print(X / N)

e = Engine(0)

move = e.best_move(board)
print(move)

board.make_move(move)

print(board._active_items)
print(board._skip_next)
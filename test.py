from buckshot_roulette.game import BuckshotRoulette
from buckshot_roulette.ai import Dealer, Random

p1_wins = 0
for _ in range(1, 100000):
    board = BuckshotRoulette(4)
    engine1 = Random(0)
    engine2 = Dealer(1)
    while board.winner() == None:
        player = engine1 if board.current_turn == 0 else engine2
        move = player.choice(board)
        res = board.make_move(move)
        player.post(move, res)
    if board.winner() == 0:
        p1_wins += 1
    print(f"[{_}] Player {board.winner()+1} Wins! {100*p1_wins / _:.2f}%")
    #print(board.items)
from cli_game import Board, Items
import itertools
from functools import lru_cache
from tqdm import tqdm

class StatEngine():
    def __init__(self, playing_as):
        self.me = playing_as
        pass
    
    def best_move(self, board: Board):
        assert board.current_turn == self.me
        
        X, N = board.shotgun_info()
        items_available = board.p1_items if self.me == 0 else board.p2_items
        
        moves = board.moves()      
        
        if 'cigarettes' in moves and board.charges[self.me] < board.max_charges:
            return 'cigarettes'
        elif 'cigarettes' in moves:
            moves.remove('cigarettes')

        if board.chamber_public == False or X == 0:
            return 'self'
        if board.chamber_public != None and 'magnifying_glass' in moves:
            moves.remove('magnifying_glass')
            

        action_pool = ['beer'] * items_available['beer']
        # Single use items
        for i in ['saw', 'handcuffs', 'magnifying_glass']:
            if i in moves:
                action_pool.append(i)
        
        actions = set()
        actions.add(())
        for i in range(1, len(action_pool)+1):
            combs = list(itertools.combinations(action_pool, i))
            for c in combs:
                valid = True
                mg = False
                for item in c:                
                    if item == 'magnifying_glass':
                        mg = True
                    if item == 'beer'and mg:
                        valid = False
                        break
                if valid:
                    lc = list(c)
                    actions.add(tuple(c))
        actions = sorted(list(actions), key=lambda x: self.evaluate_action(x, X, N, tuple(items_available.values())))
        best_action = actions[-1]
        return best_action[0] if len(best_action) > 0 else 'op'
        
    @lru_cache(1024)
    def evaluate_action(self, action, X, N, items_available: tuple):
        ex_val = self.expected_value(action, X, N) - self.expected_value((), X, N)  
        item_value = {
            'beer':0.1,
            'cigarettes':0,
            'handcuffs':0.3,
            'magnifying_glass':0.3,
            'saw':0.1
        }
        penalty = 0
        i = 0
        for key, val in item_value.items():
            if items_available[i] > 0 and key in action:
                penalty += val / items_available[i]
            i += 1
        
        return ex_val - penalty
    
    @lru_cache(1024)
    def expected_value(self, action, X, N):
        if X <= 0 or N <= 0:
            return 0
        if len(action) == 0:
            return X / N
        if isinstance(action, str):
            action = tuple([action])
        a = action[0]
        if a == 'magnifying_glass':
            see_live = 1
            if X < N:
                see_blank = self.expected_value(action[1:], X, N-1)
                return (see_live * X / N) + (see_blank * (N - X) / N)
            else:
                return 1
        elif a == 'handcuffs':
            # Handcuffs always come last
            if N < 2:
                return self.expected_value(action[1:], X, N)
            
            return 2 * self.expected_value(action[1:], X, N)
        elif a == 'saw':
            return 2 * self.expected_value(action[1:], X, N)
        elif a == 'beer':
            was_live = self.expected_value(action[1:], X-1, N-1)
            if X < N:
                was_blank = self.expected_value(action[1:], X, N-1)
                return (was_live * X / N) + (was_blank * (N - X) / N)
            else:
                return was_live     

import random
class DealerEngine():
    def __init__(self, turn):
        self.me = turn
        self.known_shell = None
        self.target = 'op'
        
    def best_move(self, board: Board):
        items = board.p1_items if self.me == 0 else board.p2_items
        item_array = []
        for i, amt in items.items():
            item_array.extend([i] * amt)

        X, N = board.shotgun_info()
        
        self.known_shell = board.chamber_public
        if N == 1:
            self.known_shell = board._shotgun[0]
        self.target = ""
        if self.known_shell != None:
            self.target = 'op' if self.known_shell == True else 'self'
        
        for item in set(item_array):
            if item == 'magnifying_glass' and self.known_shell == None and N != 1:
                return 'magnifying_glass'
            elif item == 'cigarettes' and board.charges[self.me] < board.max_charges:
                return 'cigarettes'
            elif item == 'beer' and self.known_shell == None and N != 1:
                return 'beer'
            elif item == 'handcuffs' and N != 1:
                return 'handcuffs'
            elif item == 'saw' and self.known_shell == True:
                return 'saw'
        
        self.known_shell = None
        if self.target == "":
            return random.choice(['op', 'self'])
        else:
            return self.target

class RandomEngine():
    def __init__(self, _):
        pass
    
    def best_move(self, board: Board):
        return random.choice(board.moves())

from tqdm import tqdm
import time

def run_round(board: Board, engine0, engine1):
    while board.winner() == None:
        if board.current_turn == 0:
            move = engine0.best_move(board)
            board.make_move(move)
        else:
            move = engine1.best_move(board)
            board.make_move(move)
    return board.winner()

def run_batch(engine0, engine1):
    for _ in range(3):
        board = Board(random.randint(2, 4))
        w = run_round(board, engine0, engine1)
        if w != 0:
            return 1
    return 0

if __name__ == "__main__":
    
    engine0 = StatEngine(0)
    engine1 = DealerEngine(1)
    random.seed(12345)
    wins = []
    for _ in range(10000):            
        wins.append(1 - run_batch(engine0, engine1))
    g1 = sum(wins) / len(wins)
    
    wins = []
    engine0 = DealerEngine(0)
    engine1 = StatEngine(1)
    random.seed(12345)
    for _ in range(10000):
        wins.append(run_batch(engine0, engine1))
    g2 = sum(wins) / len(wins)
    print(f"As Player: {g1}\nAs Dealer: {g2}")
        
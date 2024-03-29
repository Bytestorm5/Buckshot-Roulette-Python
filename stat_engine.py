from cli_game import Board, Items, generate_binary_numbers
import itertools
from functools import lru_cache
from tqdm import tqdm
from math import comb, sqrt

class Hypergeometric():
    @staticmethod
    def pmf(N, K, n, k):
        """
        Calculate the probability mass function (PMF) of the hypergeometric distribution.

        Parameters:
        N (int): Total number of items.
        K (int): Number of items with the feature of interest.
        n (int): Number of items drawn without replacement.
        k (int): Number of items with the feature of interest in the drawn sample.

        Returns:
        float: The probability of drawing exactly k items with the feature of interest.
        """
        if n > N:
            return 0
        try:
            return comb(K, k) * comb(N - K, n - k) / comb(N, n)
        except ZeroDivisionError:
            return 0
    @staticmethod
    def mean(N, K, n):
        """
        Calculate the mean of the hypergeometric distribution.

        Parameters:
        N (int): Total number of items.
        K (int): Number of items with the feature of interest.
        n (int): Number of items drawn without replacement.

        Returns:
        float: The mean of the hypergeometric distribution.
        """
        return n * (K / N)
    @staticmethod
    def var(N, K, n):
        """
        Calculate the variance of the hypergeometric distribution.

        Parameters:
        N (int): Total number of items.
        K (int): Number of items with the feature of interest.
        n (int): Number of items drawn without replacement.

        Returns:
        float: The variance of the hypergeometric distribution.
        """
        return n * (K / N) * (1 - K / N) * ((N - n) / (N - 1))

class StatEngine():
    def __init__(self, playing_as):
        self.me = playing_as
        pass
    def _best_move(self, board: Board):
        assert board.current_turn == self.me
        
        X, N = board.shotgun_info()
        items_available = board.items[self.me]
        
        moves = board.moves()      
        
        if 'cigarettes' in moves and board.charges[self.me] < board.max_charges:
            return 'cigarettes'
        elif 'cigarettes' in moves:
            moves.remove('cigarettes')

        if X == 0:
            # Since there's no risk to us, give the opponent every opportunity to mess up
            return 'op'
        if board.chamber_public == False:
            return 'self'        
        if board.chamber_public != None and 'magnifying_glass' in moves:
            moves.remove('magnifying_glass')        
        
        eval_dict = {}
        for i in ['handcuffs', 'magnifying_glass', 'saw', 'self', 'op']:
            if i in moves:
                eval_dict[i] = -99999
        
        eval_dict['op'] = X / N
        
        def smart_pmf(bullets_drawn, live_found):
            if live_found > bullets_drawn:
                return 0
            if bullets_drawn == 0 and live_found == 0:
                return 1
                               
            if board.chamber_public == True:
                if bullets_drawn == 1:
                    return 1 if live_found == 1 else 0                
                return Hypergeometric.pmf(N-1, X-1, bullets_drawn-1, max(live_found-1, 0))
            elif board.chamber_public == False:
                if bullets_drawn == 1:
                    return 0 if live_found == 1 else 1
                return Hypergeometric.pmf(N-1, X, bullets_drawn-1, live_found)
            else:
                return Hypergeometric.pmf(N, X, bullets_drawn, live_found)
        
        if 'magnifying_glass' in eval_dict:
            eval_dict['magnifying_glass'] = 1 - smart_pmf(2, 0)
        
        if 'saw' in eval_dict:
            if board.chamber_public == True:
                eval_dict['saw'] = 2
            else:
                eval_dict['saw'] = 2 * X / N
        
        if 'handcuffs' in eval_dict:
            if N == 1:
                eval_dict['handcuffs'] = X / N            
            elif board.chamber_public == True:
                eval_dict['handcuffs'] = 1 + (X-1) / (N-1)
            else:
                eval_dict['handcuffs'] = (2 * smart_pmf(2, 2)) + smart_pmf(2, 1)
                
        return max(eval_dict, key=eval_dict.get)
    
    
    def best_move(self, board: Board):
        assert board.current_turn == self.me
        
        X, N = board.shotgun_info()
        items_available = board.items[self.me]
        
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
        action_pool = ['self'] * (N-X)
        # Single use items
        for i in ['saw', 'magnifying_glass', 'handcuffs']:
            if i in moves:
                action_pool.append(i)
        
        actions = self.possible_actions(action_pool)
        actions = {a: self.expected_value(a, X, N) for a in actions}
                    
        if 'handcuffs' in moves:
            # Figure out followup actions
            # Oh boy
            action_pool.remove('handcuffs')
            for action in actions.keys():
                if 'handcuffs' not in action:
                    continue
                
                current_items_available = items_available.copy()
                current_pool = action_pool.copy()
                
                for i in ['saw', 'magnifying_glass']:
                    if i in action:                        
                        if items_available[i] <= 1:
                            current_pool.remove(i)  
                        current_items_available[i] -= 1
                beer_count = list(action).count('beer')
                current_items_available['beer'] -= beer_count
                for _ in range(beer_count):
                    current_pool.remove('beer')
                            
                next_actions = self.possible_actions(current_pool)
                best_live = max([self.expected_value(a, X-1, N-1) for a in next_actions])
                best_blank = max([self.expected_value(a, X, N-1) for a in next_actions])
                actions[action] += ((X/N) * best_live) + (((N-X)/N) * best_blank)
            
        best_action = max(actions, key=actions.get)
        return best_action[0] if len(best_action) > 0 else 'op'
    
    @staticmethod
    def possible_actions(action_pool: list):
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
        return actions
    
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
            return self.expected_value(action[1:], X, N)
        elif a == 'saw':
            return 2 * self.expected_value(action[1:], X, N)
        elif a == 'beer':
            was_live = self.expected_value(action[1:], X-1, N-1)
            if X < N:
                was_blank = self.expected_value(action[1:], X, N-1)
                return (was_live * X / N) + (was_blank * (N - X) / N)
            else:
                return was_live   
        elif a == 'op':
            return X / N
        elif a == 'self':
            return (-X / N) + ((N - X) / N) * self.expected_value(action[1:], X, N-1)

class NewStatEngine():
    def __init__(self, playing_as):
        self.me = playing_as
        self.op = 1 if playing_as == 0 else 0
        pass    
    def best_move(self, board: Board):
        assert board.current_turn == self.me
        
        X, N = board.shotgun_info()
        items_available = board.items[self.me]
        
        moves = board.moves()      
        
        if 'cigarettes' in moves and board.charges[self.me] < board.max_charges:
            return 'cigarettes'
        elif 'cigarettes' in moves:
            moves.remove('cigarettes')

        known_bullet = board.chamber_public
        if X == 0:
            known_bullet = False
        if X == N:
            known_bullet = True
        
        if known_bullet == False:
            return 'self'        
        if known_bullet == True:
            if X == N:
                if 'handcuffs' in moves and X >= 2:
                   if X > 2:
                       # Deny opponent as many lives as possible
                       return 'beer' if 'beer' in moves else 'handcuffs'
                   else:
                       return 'handcuffs'                   
                else:
                    if 'saw' in moves:
                        return 'saw'
                    else:
                        return 'op'
            elif 'saw' in moves:
                return 'saw'
        
        if known_bullet != None and 'magnifying_glass' in moves:
            moves.remove('magnifying_glass')
        if (X < 2 or N < 2) and 'handcuffs' in moves:
            moves.remove('handcuffs')
        

        action_pool = ['beer'] * items_available['beer']
        action_pool += ['self']
        # Single use items
        for i in ['magnifying_glass', 'saw', 'handcuffs']:
            if i in moves:
                action_pool.append(i)
        
        dmg = self.get_dmg(board)
        dmg = (min(dmg[0], board.charges[1]), min(dmg[1], board.charges[0]))
        
        hit_chance = X / N
        if known_bullet == True:
            hit_chance = 1
        elif known_bullet == False:
            hit_chance = 0
        
        actions = self.possible_actions(action_pool) #, board.items[self.me]['beer'])
        actions = {a: self.expected_value(a, X, N, dmg, hit_chance) for a in actions}
                    
        if 'handcuffs' in moves:
            # Figure out followup actions
            # Oh boy
            action_pool.remove('handcuffs')
            for action in actions.keys():
                if 'handcuffs' not in action:
                    continue
                
                current_items_available = items_available.copy()
                current_pool = action_pool.copy()
                
                for i in ['saw', 'magnifying_glass']:
                    if i in action:                        
                        if items_available[i] <= 1:
                            current_pool.remove(i)  
                        current_items_available[i] -= 1
                beer_count = list(current_pool).count('beer')
                current_items_available['beer'] -= beer_count
                for _ in range(beer_count):
                    current_pool.remove('beer')
                            
                next_actions = self.possible_actions(current_pool)
                best_live = max([self.expected_value(a, X-1, N-1, dmg) for a in next_actions])
                best_blank = max([self.expected_value(a, X, N-1, dmg) for a in next_actions])
                actions[action] += ((X/N) * best_live) + (((N-X)/N) * best_blank)

        best_action = None
        best_value = float('-inf')
        for action, val in actions.items():
            if val > best_value or \
                (val == best_value and len(action) < len(best_action)):
                    
                best_action = action
                best_value = val            
                
        return best_action[0] if len(best_action) > 0 else 'op'
    
    def get_dmg(self, board: Board):
        """Worst-case scenario- how much damage can the opponent do

        Args:
            board (Board): _description_
        """
        X, N = board.shotgun_info()
        me_dmg = 1
        op_dmg = -1
        if board.items[self.op]['saw'] > 0:
            op_dmg = -2
        if board.items[self.op]['handcuffs'] > 0 and X > 1:
            if board.items[self.op]['saw'] == 1:
                op_dmg = -3
            elif board.items[self.op]['saw'] > 1:
                op_dmg = -4
            else:
                op_dmg = -2        
        
        if board._active_items['saw'] > 0:
            me_dmg = 2
        return me_dmg, op_dmg
    
    @staticmethod
    def possible_actions(action_pool: list):
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
                    actions.add(tuple(c))
        return actions
    
    @lru_cache(1024)
    def expected_value(self, action, X, N, dmg: tuple=None, hit_chance = None):
        if X <= 0 or N <= 0 or X > N:
            return 0
        
        if hit_chance == None:
            hit_chance = X / N
        if X == 0:
            hit_chance = 0
        elif X == N:
            hit_chance = 1
        
        if len(action) == 0:
            return (dmg[0] * hit_chance) + (dmg[1] if N > 1 and X > 0 else 0)
        if isinstance(action, str):
            action = tuple([action])
        a = action[0]
        
        if a == 'magnifying_glass':
            see_live = self.expected_value(action[1:], X, N, dmg, 1)
            if hit_chance < 1 and X < N:
                # Guaranteed Skip
                see_blank = self.expected_value(action[1:], X, N-1, dmg)
                return (see_live * hit_chance) + (see_blank * (1-hit_chance))
            else:
                return 1
        elif a == 'handcuffs':
            return self.expected_value(action[1:], X, N, dmg, hit_chance) - dmg[1]
        
        elif a == 'saw':
            n_dmg = (dmg[0] * 2, dmg[1])
            return self.expected_value(action[1:], X, N, n_dmg, hit_chance)
        elif a == 'beer':
            was_live = self.expected_value(action[1:], X-1, N-1, dmg)
            if hit_chance < 1 and X < N:
                was_blank = self.expected_value(action[1:], X, N-1, dmg)
                return (was_live * hit_chance) + (was_blank * (1-hit_chance))
            else:
                return was_live   
        elif a == 'op':
            return (dmg[0] * hit_chance) + (dmg[1] * (X-1) / (N-1) if N > 1 and X > 0 else 0)
        elif a == 'self':
            is_hit = -dmg[0] * hit_chance
            if X > 1:
                is_hit += dmg[1] * (X-1) / (N-1)
            
            is_miss = self.expected_value(action[1:], X, N-1, dmg) * (1 - hit_chance)
            if X > 1 and N > 2:
                is_miss += dmg[1] * (X-1) / (N-2)            
            
            return is_hit + is_miss

import random
class DealerEngine():
    def __init__(self, turn):
        self.me = turn
        self.known_shell = None
        self.target = 'op'
        
    def best_move(self, board: Board):
        items = board.items[self.me]
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

from new_engine import expectimax
class NewEngine():
    def __init__(self, me):
        self.me = me
    
    def best_move(self, board: Board):
        return expectimax(board, 8, self.me)[0]

class CheatEngine2():
    def __init__(self, turn):
        self.me = turn
    def best_move(self, board: Board):
        moves = board.moves()
        X, N = board.shotgun_info()
        
        if 'cigarettes' in moves and board.charges[self.me] < board.max_charges:
            return 'cigarettes'

        if 'beer' in moves:
            beer_count = board.items[self.me]['beer']
            min_X = float('inf')
            min_c = 0
            for c in range(0, beer_count):
                mX = sum([1 if x else 0 for x in board._shotgun[c:]])
                if mX < min_X and X > 0:
                    min_X = mX
                    min_c = c
            if min_c > 0:
                return 'beer'
        
        if 'magnifying_glass' in moves:
            return 'magnifying_glass'
        
        if board._shotgun[0] == False:
            return 'self'
        else:
            if board.items[self.me]['saw'] > 0:
                return 'saw'
            elif X > 2 and 'handcuffs' in moves:
                return 'handcuffs'
            else:
                return 'op'
            

class CheatEngine():
    def __init__(self, turn):
        self.me = turn
    
    @lru_cache(100000)
    def heuristic_value(self, board: Board) -> float:
        # Implement a heuristic function to evaluate board states
        op = 1 if self.me == 0 else 0
        
        if board.winner() == self.me:
            return 1 + board.max_charges * 2
        if board.winner() == op:
            return -(1 + board.max_charges * 2)
        
        h = board.charges[self.me] - board.charges[op]       
        h = h / max(len(board._shotgun), 1)
        
        if board.chamber_public != None:
            h += 0.01       
        #h += 0.01 * (sum(board.items[self.me].values()) - sum(board.items[op].values()))
        return h
    
    @lru_cache(10000)
    def minimax(self, board: Board, depth, maximizingPlayer):
        if depth == 0 or board.winner() is not None or len(board._shotgun) == 0:
            return self.heuristic_value(board)

        if maximizingPlayer:
            maxEval = float('-inf')
            for move in board.moves():
                new_depth = depth
                if move == 'op' or (move == 'self' and board._shotgun[0] == True):
                    new_depth -= 1
                
                new_board = board.copy()
                new_board.make_move(move, load_new=False)
                eval = self.minimax(new_board, new_depth, self.me == new_board.current_turn)
                maxEval = max(maxEval, eval)
            return maxEval
        else:
            minEval = float('inf')
            for move in board.moves():
                new_depth = depth
                if move == 'op' or (move == 'self' and board._shotgun[0] == True):
                    new_depth -= 1
                    
                new_board = board.copy()
                new_board.make_move(move, load_new=False)
                eval = self.minimax(new_board, new_depth, self.me == new_board.current_turn)
                minEval = min(minEval, eval)
            return minEval

    @lru_cache(1024)
    def best_move(self, board: Board, depth=8, is_root=True):
        best_value = float('-inf') if self.me == board.current_turn else float('inf')
        best_move = None

        if is_root and 'cigarettes' in board.moves() and board.charges[self.me] < board.max_charges:
            return 'cigarettes'
                
        for move in board.moves():
            new_depth = depth
            if move == 'op' or (move == 'self' and board._shotgun[0] == True):
                new_depth -= 1
                    
            new_board = board.copy()
            new_board.make_move(move, load_new=False)
            value = self.minimax(new_board, new_depth, self.me == new_board.current_turn)

            if self.me == board.current_turn and value > best_value:
                best_value = value
                best_move = move
            elif self.me != board.current_turn and value < best_value:
                best_value = value
                best_move = move

        return best_move if is_root else best_value
            
class UnCheatEngine():
    def __init__(self, turn):
        self.me = turn
        self.engine = CheatEngine2(self.me)
    def best_move(self, board: Board, depth=8):
        X, N = board.shotgun_info()
        best_dict = {}
        for possibility in generate_binary_numbers(X, N):
            new_board = board.copy()
            new_board._shotgun = possibility
            
            move = self.engine.best_move(new_board)
            if move in best_dict:
                best_dict[move] += 1
            else:
                best_dict[move] = 1
        cost = {
            'op': 0,
            'self': 0.5,
            'handcuffs':0.2,
            'magnifying_glass':0.2,
            'beer':0.1,
            'cigarettes':-0.1,
            'saw':0.15
        }
        for key, val in cost.items():
            if key in best_dict:
                best_dict[key] -= val
        if len(board.moves()) > 2:
            pass
        return max(best_dict, key=best_dict.get)
            

from tqdm import tqdm
import time

def heuristic(board: Board):
    h = board.charges[0] - board.charges[1]
    return h / max(board.charges)

def run_round(board: Board, engine0, engine1):
    while board.winner() == None:
        if board.current_turn == 0:
            # board.items[0]['saw'] = 0
            # board.items[1]['saw'] = 0
            move = engine0.best_move(board)
            board.make_move(move)
        else:
            # board.items[0]['saw'] = 0
            # board.items[1]['saw'] = 0
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
    import time
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np
    iterations = 10000
    
    # for lives in range(1, 5):
    #     random.seed(12345)
    #     wins = []
    #     for _ in range(iterations):
    #         engine0 = NewStatEngine(0)
    #         engine1 = DealerEngine(1)
            
    #         wins.append(1 - run_round(Board(lives), engine0, engine1))
    #     g1 = sum(wins) / len(wins)
    #     g1_std = sqrt((g1 * (1 - g1)) / iterations)
    #     print(f"[{lives}]: {g1}, SE: {g1_std:.5f}")
    
    in_batches = False
    
    if in_batches:
        engine0 = NewStatEngine(0)
        engine1 = DealerEngine(1)
        wins = []
        for _ in range(iterations):
            winner = run_batch(engine0, engine1)
            wins.append(1 - winner)
        g1 = sum(wins) / len(wins)
        
        g1_std = sqrt((g1 * (1 - g1)) / iterations)
        print(str(g1) + "\t", end="")    
    else:
        for lives in range(1, 5):
            for bullets in range(2, 9):
                engine0 = NewStatEngine(0)
                engine1 = DealerEngine(1)
                random.seed(12345)
                wins = []
                for _ in range(iterations):
                    winner = run_round(Board(lives, bullets), engine0, engine1)
                    wins.append(1 - winner)
                g1 = sum(wins) / len(wins)
                
                g1_std = sqrt((g1 * (1 - g1)) / iterations)
                print(str(g1) + "\t", end="")
            print()

            #print(f"Lives: {lives}\nAs Player: {g1*100}% std: {g1_std*100}%\n")
        
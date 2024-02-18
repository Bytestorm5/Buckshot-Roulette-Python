from cli_game import Board, Items
from functools import lru_cache

cache = {}
def cache_max(board: Board, depth: int, eval_for: int) -> tuple[str, float]:
    X, N = board.shotgun_info() 
    key = (X, N, board.items[0], board.items[1], board._active_items, board.chamber_public, board.charges[0], board.charges[1], depth, eval_for, board.current_turn, board._skip_next)
    if key in cache:
        return cache[key]
    else:
        val = expectimax(board, depth, eval_for)
        cache[key] = val
        return val

@lru_cache(100000)
def expectimax(board: Board, depth: int, eval_for: int) -> tuple[str, float]:  
    X, N = board.shotgun_info() 
    if depth == 0 or board.winner() is not None or N == 0:
        return None, heuristic_value(board, eval_for)

    moves = board.moves()      
    
    if 'cigarettes' in moves:
        if board.charges[board.current_turn] < board.max_charges:
            return 'cigarettes', float('inf')
        else:
            moves.remove('cigarettes')
    if board.chamber_public != False and X > 0:
        moves.remove('self')
    if (board.chamber_public != None or N < 2) and 'magnifying_glass' in moves:
        moves.remove('magnifying_glass')
    if (board.charges[board.opponent()] == 1 or board._active_items['saw'] > 0) and 'saw' in moves:
        moves.remove('saw')
    if N < 2 and 'handcuffs' in moves:
        moves.remove('handcuffs')
        
    if len(moves) == 1:
        return moves[0], 0
    
    def switch(b: Board, is_hit: bool, at_opponent: bool):
        b._active_items['saw'] = 0
        b.chamber_public = None
        if not at_opponent and not is_hit:
            return
        if b._active_items['handcuffs'] > 0.5:            
            b._active_items['handcuffs'] -= 0.5
        
        if b._skip_next:
            b._skip_next = False
        else:
            b.switch_turn()
    
    max_util = float('-inf')
    max_move = 'op'
    for move in moves:  
        utility = 0    
        hit_prob = X / N
        if board.chamber_public == True:
            hit_prob = 1.0
        elif board.chamber_public == False:
            hit_prob = 0.0  
        # if X == N:
        #     hit_prob = 1
        # elif X == 0:
        #     hit_prob = 0
        
        
        if move == 'op': 
            op_util = 0          
            if hit_prob > 0:
                hit_board = board.copy()
                hit_board._shotgun.remove(True)
                hit_board.charges[board.opponent()] -= 2 if board._active_items['saw'] > 0 else 1
                switch(hit_board, True, True)
                
                op_util += hit_prob * expectimax(hit_board, depth-1, eval_for)[1]
            
            if hit_prob < 1:
                miss_board = board.copy()
                miss_board._shotgun.remove(False)
                switch(miss_board, False, True)
                
                op_util += (1 - hit_prob) * expectimax(miss_board, depth-1, eval_for)[1]
            
            utility += op_util if eval_for == 0 else 0.5 * op_util
        elif move == 'self':            
            self_util = 0
            if hit_prob > 0:
                hit_board = board.copy()
                hit_board._shotgun.remove(True)
                hit_board.charges[board.current_turn] -= 1
                switch(hit_board, True, False)
                
                self_util += hit_prob * expectimax(hit_board, depth-1, eval_for)[1]
            
            if hit_prob < 1:
                miss_board = board.copy()
                miss_board._shotgun.remove(False)
                switch(miss_board, False, False)
                
                self_util += (1 - hit_prob) * expectimax(miss_board, depth-1, eval_for)[1]
            
            utility += self_util if eval_for == 0 else 0.5 * self_util
        elif move == 'magnifying_glass':
            board.items[board.current_turn]['magnifying_glass'] -= 1
            if hit_prob > 0:
                hit_board = board.copy()
                hit_board.chamber_public = True
                
                utility += hit_prob * expectimax(hit_board, depth, eval_for)[1]
            
            if hit_prob < 1:
                miss_board = board.copy()
                miss_board.chamber_public = False
                
                utility += (1 - hit_prob) * expectimax(miss_board, depth, eval_for)[1]
        elif move == 'beer':
            board.items[board.current_turn]['beer'] -= 1
            board.chamber_public = None
            if hit_prob > 0:
                hit_board = board.copy()
                hit_board._shotgun.remove(True)
                
                utility += hit_prob * expectimax(hit_board, depth, eval_for)[1]
            
            if hit_prob < 1:
                miss_board = board.copy()
                miss_board._shotgun.remove(False)
                
                utility += (1 - hit_prob) * expectimax(miss_board, depth, eval_for)[1]
        elif move == 'saw':
            board.items[board.current_turn]['saw'] -= 1
            new_board = board.copy()
            new_board._active_items['saw'] += 1
            _, utility = expectimax(new_board, depth, eval_for)
        elif move == 'handcuffs':
            board.items[board.current_turn]['handcuffs'] -= 1
            new_board = board.copy()
            new_board._active_items['handcuffs'] += 1
            _, utility = expectimax(new_board, depth, eval_for)
        
        if utility > max_util:
            max_util = utility
            max_move = move
    
    return max_move, max_util
    

def heuristic_value(board: Board, eval_for) -> float:
    # Implement a heuristic function to evaluate board states
    op = 1 if eval_for == 0 else 0
    
    h = board.charges[eval_for] - board.charges[op]    
    h -= 0.01 * (sum(board.items[eval_for].values()) - sum(board.items[op].values()))
    return h

#board = Board(charge_count=5)
#best_utility = expectimax(board, depth=3, is_chance_node=False, alpha=float('-inf'), beta=float('inf'))
from game import BuckshotRoulette
from typing import Literal
from dataclasses import asdict
import random
    
class Dealer:
    def __init__(self, playing_as: Literal[0, 1]):        
        self.known_shells: list[bool] = None
        self.target = None
        self.known_shell = None
        
    def _figure_out_shell(self, board: BuckshotRoulette):
        if self.known_shells[0]:
            return True
        
        c_live, c_blank = board.shotgun_info()
        c_blank -= c_live
        
        if c_live == 0 or c_blank == 0:
            return True
        
        for i in range(len(self.known_shells)):
            if self.known_shells[i]:
                if board._shotgun[i]:
                    c_live -= 1
                else:
                    c_blank -= 1
        
        if c_live == 0 or c_blank == 0:
            return True
        
        return False        
       
    def make_move(self, board: BuckshotRoulette):
        if self.known_shells == None:
            self.known_shells = [False] * len(board._shotgun)
            self.known_shells[-1] = True
        
        if self.known_shell == None:
            if self._figure_out_shell(board):
                self.known_shells[0] = True
                if board._shotgun[0]:
                    self.known_shell = True
                    self.target = 'op'
                else:
                    self.known_shell = False
                    self.target = 'self'
        
        if len(board._shotgun) == 1:
            self.known_shell = board._shotgun[0]
            self.target = 'op' if self.known_shell else 'self'
        
        using_adrenaline = board.items[self.me].adrenaline > 0
        
        has_cigs = board.items[self.me].cigarettes > 0
        wants_to_use = None
        
        moves = board.legal_items()
        for item in board.POSSIBLE_ITEMS:
            if board.items[0][item] < 1 and board.items[1][item] < 1:
                moves.remove(item)
                
        for item in moves:
            if item == 'magnifying_glass' and self.known_shell == None and len(board._shotgun) != 1:
                wants_to_use = item
                break
            if item == 'cigarettes' and board.charges[self.me] < board.max_charges:
                wants_to_use = item
                has_cigs = False
                break
            if item == 'meds' and board.charges[1 - self.me] < board.max_charges and not has_cigs and not using_medicine:
                wants_to_use = item
                using_medicine = True
                break
            if item == 'beer' and self.known_shell != True and len(board._shotgun) != 1:
                wants_to_use = item
                self.known_shell = None
                break
            if item == 'handcuffs' and len(board._shotgun) != 1:
                wants_to_use = item
                break
            if item == 'saw' and self.known_shell == True:
                wants_to_use = item
                break
            if item == 'burner_phone' and len(board._shotgun) > 2:
                wants_to_use = item
                break
            if item == 'inverter' and self.known_shell == False:
                wants_to_use = item
                self.known_shell = True                
                break
        
        main_loop_finished = wants_to_use == None
        has_saw = board.items[self.me].saw > 0
        if main_loop_finished and 'saw' in moves and self.known_shell == True:
            decision = random.random() > 0.5
            if decision:
                self.target = 'self'
            else:
                self.target = 'op'
                wants_to_use = 'saw'
        if wants_to_use != None:
            if using_adrenaline and board.items[self.me][wants_to_use] < 1:
                board.make_move('adrenaline')
            result = board.make_move(wants_to_use)
            match wants_to_use:
                case 'magnifying_glass':
                    self.known_shells[0] = True
                    self.known_shell = board._shotgun[0]
                    pass
                case 'beer':
                    self.known_shells = self.known_shells[1:]
                    pass
                case 'burner_phone':
                    self.known_shells[result[0]] = True
        else:
            if self.target == None:
                decision = random.random() > 0.5
                if decision:
                    board.make_move('op')
                else:
                    board.make_move('self')
            else:
                board.make_move(self.target)
                self.target = None
            self.known_shells = self.known_shells[1:]
            self.known_shell = board._shotgun[0] if self.known_shells[0] else None
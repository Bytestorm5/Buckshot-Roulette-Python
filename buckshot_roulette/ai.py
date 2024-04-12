from buckshot_roulette.game import BuckshotRoulette
from typing import Literal
from dataclasses import asdict
import random
import abc
import typing

class AbstractEngine(abc.ABC):
    def __init__(self, playing_as: Literal[0, 1]):
        self.me = playing_as
    
    @abc.abstractmethod
    def choice(self, board:BuckshotRoulette):
        """Determines what move this model takes from this given board position

        Args:
            board (BuckshotRoulette): The current game state
        """
        pass
    
    @abc.abstractmethod
    def post(self, last_move, result):
        """Any post-processing steps that the model needs to make after a move has been made. Typically used to store results of the magnifying glass and burner phone.

        Args:
            last_move (str): The move that the engine just made
            result (Any): The output of board.make_move
        """
        pass

class Dealer(AbstractEngine):
    def __init__(self, playing_as: Literal[0, 1]):   
        self.me = playing_as     
        self.known_shells: list[bool] = None
        self.target = None
        self.known_shell = None
        
    def _figure_out_shell(self, board: BuckshotRoulette):
        while len(self.known_shells) > len(board._shotgun):
            self.known_shells = self.known_shells[1:]
        while len(self.known_shells) < len(board._shotgun):
            self.known_shells.append(False)
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
       
    def choice(self, board: BuckshotRoulette):
        if self.known_shells == None:
            self.known_shells = [False] * len(board._shotgun)
            self.known_shells[-1] = True
        list_diff = len(self.known_shells) - len(board._shotgun)
        if list_diff < 0:
            self.known_shells.extend([False] * -list_diff)
        if list_diff > 0:
            # Should basically never be a difference > 1
            self.known_shells = self.known_shells[list_diff:]
        
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
        
        using_medicine = None
        wants_to_use = None
        
        moves = board.legal_items()
        for item in board.POSSIBLE_ITEMS:
            if item in moves and board.items[self.me][item] < 1 and (board.items[self.me].adrenaline < 1 or board.items[1][item] < 1):
                moves.remove(item)
            
        for item in moves:
            if item == 'magnifying_glass' and self.known_shell == None and len(board._shotgun) != 1:
                wants_to_use = item
                break
            if item == 'cigarettes' and board.charges[self.me] < board.max_charges:
                wants_to_use = item
                break
            if item == 'meds' and board.charges[1 - self.me] < board.max_charges and not 'cigarettes' in moves and not using_medicine:
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

        if wants_to_use == None and 'saw' in moves and self.known_shell == True:
            decision = random.random() > 0.5
            if decision:
                self.target = 'self'
            else:
                self.target = 'op'
                wants_to_use = 'saw'
        if wants_to_use != None:
            if 'adrenaline' in moves and board.items[self.me][wants_to_use] < 1:
                return 'adrenaline'
            return wants_to_use        
        else:
            if self.target == None:
                decision = random.random() > 0.5
                if decision:
                    return 'op'
                else:
                    return 'self'
            else:
                return self.target
    
    def post(self, last_move, move_result):        
        if last_move in ['op', 'self']:
            self.target = None
            
        match last_move:
            case 'magnifying_glass':
                self.known_shells[0] = True
                self.known_shell = move_result
                pass
            case 'burner_phone':
                self.known_shells[move_result[0]] = True
        
class Random(AbstractEngine):
    def __init__(self, playing_as):
        pass
    
    def choice(self, board: BuckshotRoulette):
        return random.choice(board.moves())

    def post(self, last_move, res):
        pass
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
    
    def shell_at(self, idx, board: BuckshotRoulette) -> Literal[True, False, None]:
        if self.known_shells[idx]:
            return board._shotgun[idx]
        else:
            live = sum([int(x) for x in board._shotgun])
            blank = len(board._shotgun) - live
            
            if live == 0:
                return False
            elif blank == 0:
                return True
            
            for i in range(len(self.known_shells)):
                if self.known_shells[i]:
                    if board._shotgun[i]:
                        live -= 1
                    else:
                        blank -= 1
            
            if live == 0:
                return False
            elif blank == 0:
                return True
            
            return None        
    
    def choice(self, board: BuckshotRoulette):
        if self.known_shells == None:
            self.known_shells = [False] * len(board._shotgun)
            self.known_shells[-1] = True
        while len(self.known_shells) > len(board._shotgun):
            self.known_shells = self.known_shells[1:]
        while len(self.known_shells) < len(board._shotgun):
            self.known_shells.append(False)
        
        moves = board.moves()
        own_moves = moves.copy()
        if 'adrenaline' in moves:
            for item in board.items[1 - self.me]:
                if item not in moves:
                    moves.append(item)
        wants_to_use = None
        
        using_medicine = False
        
        for item in moves:
            if item == 'magnifying_glass' and not self.known_shells[0] and len(board._shotgun) != 1:
                wants_to_use = item
                break
            if item == 'cigarettes' and board.charges[self.me] < board.max_charges:
                wants_to_use = item
                break
            if item == 'meds' and board.charges[1 - self.me] < board.max_charges and not 'cigarettes' in moves and not using_medicine:
                wants_to_use = item
                using_medicine = True
                break
            if item == 'beer' and self.shell_at(0, board) != True and len(board._shotgun) != 1:
                wants_to_use = item
                break
            if item == 'handcuffs' and len(board._shotgun) != 1:
                wants_to_use = item
                break
            if item == 'saw' and self.shell_at(0, board) == True:
                wants_to_use = item
                break
            if item == 'burner_phone' and len(board._shotgun) > 2:
                wants_to_use = item
                break
            if item == 'inverter' and self.shell_at(0, board) == False:
                wants_to_use = item        
                break
        
        if wants_to_use == None:
            if len(board._shotgun) == 1 or self.shell_at(0, board) != None:
                if board._shotgun[0]:
                    return 'op'
                else:
                    return 'self'
            else:
                return 'op' if random.random() > 0.5 else 'self'
        else:
            if wants_to_use in own_moves:
                return wants_to_use
            elif 'adrenaline' in own_moves:
                return 'adrenaline'
            else:
                # Should never happen
                raise RuntimeError("Attempted invalid move without adrenaline")
    
    def post(self, last_move, move_result):        
        match last_move:
            case 'op', 'self':
                self.known_shells = self.known_shells[1:] if len(self.known_shells) > 0 else []
                pass
            case 'magnifying_glass':
                self.known_shells[0] = True
            case 'burner_phone':
                self.known_shells[move_result[0]] = True
            
            
        
class Random(AbstractEngine):
    def __init__(self, playing_as):
        pass
    
    def choice(self, board: BuckshotRoulette):
        return random.choice(board.moves())

    def post(self, last_move, res):
        pass
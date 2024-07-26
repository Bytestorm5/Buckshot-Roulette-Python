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
        self.last_shell = None
    
    def shell_at(self, idx, board: BuckshotRoulette) -> Literal[True, False, None]:
        if self.known_shells[idx] != None:
            return self.known_shells[idx]
        else:
            live = board.live
            blank = board.total - board.live
            
            if live == 0:
                return False
            elif blank == 0:
                return True
            
            for i in range(len(self.known_shells)):
                if self.known_shells[i] != None:
                    if self.known_shells[i]:
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
            self.known_shells = [None] * board.total     
            # The dealer always knows the last shell
            self.known_shells[-1] = self.last_shell    
            if board.total == 1:
                self.known_shells[0] = board.live > 0
        while len(self.known_shells) > board.total:
            self.known_shells = self.known_shells[1:]
        while len(self.known_shells) < board.total:
            self.known_shells.append(None)
        
        moves = board.moves()
        own_moves = moves.copy()
        if 'adrenaline' in moves:
            for item in board.items[1 - self.me]:
                if item not in moves:
                    moves.append(item)
        wants_to_use = None
        
        using_medicine = False
        
        for item in moves:
            if item == 'magnifying_glass' and not self.known_shells[0] and board.total != 1:
                wants_to_use = item
                break
            if item == 'cigarettes' and board.charges[self.me] < board.max_charges:
                wants_to_use = item
                break
            if item == 'meds' and board.charges[1 - self.me] < board.max_charges and not 'cigarettes' in moves and not using_medicine:
                wants_to_use = item
                using_medicine = True
                break
            if item == 'beer' and self.shell_at(0, board) != True and board.total != 1:
                wants_to_use = item
                break
            if item == 'handcuffs' and board.total != 1:
                wants_to_use = item
                break
            if item == 'saw' and self.shell_at(0, board) == True:
                wants_to_use = item
                break
            if item == 'burner_phone' and board.total > 2:
                wants_to_use = item
                break
            if item == 'inverter' and self.shell_at(0, board) == False:
                wants_to_use = item        
                break
        
        if wants_to_use == None:
            if board.total == 1 or self.shell_at(0, board) != None:
                if self.shell_at(0, board):
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
                self.known_shells[0] = move_result
            case 'burner_phone':
                self.known_shells[move_result[0]] = move_result[1]
            
            
        
class Random(AbstractEngine):
    def __init__(self, playing_as):
        pass
    
    def choice(self, board: BuckshotRoulette):
        return random.choice(board.moves())

    def post(self, last_move, res):
        pass

class Human(AbstractEngine):
    def __init__(self, playing_as, knowledge = True):
        self.knowledge = knowledge
        self.known_shells: list[bool] = None
    
    def shell_at(self, idx, board: BuckshotRoulette) -> Literal[True, False, None]:
        if self.known_shells == None:
            self.known_shells = [None] * board.total
            if board.total == 1:
                self.known_shells[0] = board.live > 0
        while len(self.known_shells) > board.total:
            self.known_shells = self.known_shells[1:]
        while len(self.known_shells) < board.total:
            self.known_shells.append(None)
        
        if self.known_shells[idx] != None:
            return self.known_shells[idx]
        else:
            live = board.live
            blank = board.total - board.live
            
            if live == 0:
                return False
            elif blank == 0:
                return True
            
            for i in range(len(self.known_shells)):
                if self.known_shells[i] != None:
                    if self.known_shells[i]:
                        live -= 1
                    else:
                        blank -= 1
            
            if live == 0:
                return False
            elif blank == 0:
                return True
            
            return None
    
    def choice(self, board: BuckshotRoulette):
        selfhealth = board.charges[board.current_turn]
        opphealth = board.charges[1-board.current_turn]
        moves = board.moves()
        
        print(f'''\
your move.
charges:
self   : other
{'🗲'*selfhealth + ' '*(6-selfhealth)} : {'🗲'*opphealth + ' '*(6-opphealth)}

items (self):
{board.items[board.current_turn]}

items (enemy):
{board.items[1-board.current_turn]}

active items:
{board._active_items}

{board.total} bullets left
''')
        if self.knowledge:
            print(f"{board.live} live. {board.total-board.live} blank")
            shells = [self.shell_at(i, board) for i in range(board.total)]
            shells = [{True: "L", False: "B", None: "?"}[x] for x in shells]
            print("".join(shells))
        print(f"moves:\n{moves}")
        chosen = ""
        while chosen not in moves:
            chosen = input("Move:").strip()
        
        return chosen

    def post(self, last_move, move_result):
        #print(f"result: {move_result}")
        match last_move:
            case 'op', 'self':
                self.known_shells = self.known_shells[1:] if len(self.known_shells) > 0 else []
            case 'magnifying_glass':
                self.known_shells[0] = move_result
            case 'burner_phone':
                self.known_shells[move_result[0]] = move_result[1]
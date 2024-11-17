from buckshot_roulette.multiplayer.game import BuckshotRoulette, GameStatus
from typing import Literal, Union, Tuple, List, Optional
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
    def on_own_move(self, last_move, result):
        """Any post-processing steps that the model needs to make after a move has been made. Typically used to store results of the magnifying glass and burner phone.

        Args:
            last_move (str): The move that the engine just made
            result (Any): The output of board.make_move
        """
        pass
    
    @abc.abstractmethod
    def on_opponent_move(self, last_move, result):
        """Any post-processing steps that the model needs to make after a move has been made. Typically used to store results of the magnifying glass and burner phone.

        Args:
            last_move (str): The move that the engine just made
            result (Any): The output of board.make_move
        """
        pass
    
    @abc.abstractmethod
    def on_reload(self, board:BuckshotRoulette):
        """Any internal steps to perform on a reload (like resetting knowledge)
        
        Args:
            board (BuckshotRoulette): The new game state
        """

class Dealer(AbstractEngine):
    def __init__(self, playing_as: int):
        """
        Initialize the Dealer.

        Args:
            playing_as (int): The index of the player (e.g., 0, 1, 2, 3).
        """
        self.me = playing_as
        self.known_shells: List[Optional[bool]] = []
        self.last_shell: Optional[bool] = None

    def shell_at(self, idx: int, board: 'BuckshotRoulette') -> Optional[bool]:
        """
        Determine the state of the shell at a given index.

        Args:
            idx (int): The index of the shell.
            board (BuckshotRoulette): The current game state.

        Returns:
            Optional[bool]: True if live, False if blank, None if unknown.
        """
        if idx < len(self.known_shells) and self.known_shells[idx] is not None:
            return self.known_shells[idx]
        else:
            live = board.live
            blank = board.total - board.live

            # Recalculate based on known shells
            for known in self.known_shells:
                if known is not None:
                    if known:
                        live -= 1
                    else:
                        blank -= 1

            if live == 0:
                return False
            elif blank == 0:
                return True

            return None

    def choice(self, game: BuckshotRoulette) -> Tuple[str, int]:
        """
        Decide the next move based on the current game state.

        Args:
            game (BuckshotRoulette): The current game state.

        Returns:
            Union[str, Tuple[int, str]]: The chosen move, which can be a string (e.g., 'saw')
            or a tuple representing an action on an opponent's item (e.g., (1, 'jammer')).
        """
        # Initialize known_shells if empty
        if not self.known_shells:
            self.known_shells = [None] * game.total
            if game.total == 1:
                self.known_shells[0] = game.live > 0

        # Adjust known_shells length based on current total shells
        if len(self.known_shells) > game.total:
            self.known_shells = self.known_shells[-game.total:]
        elif len(self.known_shells) < game.total:
            self.known_shells += [None] * (game.total - len(self.known_shells))

        # Retrieve available moves
        moves = game.moves()
        
        wants_to_use = None
        wants_to_target = 0
        # Prioritize item usage based on strategy
        for move in moves:
            # Handle both string and tuple moves
            target = move[0]
            item = move[1]
            
            if item == 'magnifying_glass' and not self.shell_at(0, game) and game.total != 1:
                wants_to_use = item
                wants_to_target = target
                break
            if item == 'cigarettes' and game.charges[self.me] < game.config.start_charges:
                wants_to_use = item
                wants_to_target = target
                break            
            if item.startswith('jammer') and game.total > 1 and GameStatus(int(item.split('_')[1])) not in game.statuses:
                wants_to_use = item
                wants_to_target = target
                break
            if item == 'remote':
                wants_to_use = item
                wants_to_target = target
                break
            if item == 'saw' and self.shell_at(0, game) == True:
                wants_to_use = item
                wants_to_target = target
                break
            if item == 'burner_phone' and game.total > 2:
                wants_to_use = item
                wants_to_target = target
                break
            if item == 'beer' and self.shell_at(0, game) != True and game.total != 1:
                wants_to_use = item
                wants_to_target = target
                break
            if item == 'inverter' and self.shell_at(0, game) == False:
                wants_to_use = item
                wants_to_target = target
                break

        # Decide to use an item or take a shot
        if wants_to_use is None:
            # Decide between shooting or other actions based on shell knowledge
            if game.total == 1 or self.shell_at(0, game) is not None:
                if self.shell_at(0, game):
                    # Prefer shooting an opponent if the shell is live
                    target = self.select_opponent_to_shoot(game)
                    return f'shoot_{target}', None
                else:
                    # If shell is known to be blank, take a safe action
                    # Here, we can choose to shoot ourselves or perform another safe action
                    return 'shoot_0', None  # Equivalent to 'self'
            else:
                # If shell is unknown, decide randomly
                target = self.select_opponent_to_shoot(game)
                return (f'shoot_{target}' if random.random() > 0.5 else 'shoot_0'), None
        else:
            # Execute the desired move
            if isinstance(wants_to_use, str):
                return wants_to_use, wants_to_target
            elif isinstance(wants_to_use, tuple):
                return wants_to_use, wants_to_target  # e.g., (target_idx, 'jammer')
            else:
                # Fallback in case of unexpected move type
                return 'shoot_0', None

    def select_opponent_to_shoot(self, game: 'BuckshotRoulette') -> int:
        """
        Select an opponent to shoot based on some strategy.
        Currently selects a random living opponent.

        Args:
            game (BuckshotRoulette): The current game state.

        Returns:
            int: The index of the opponent to shoot.
        """
        opponents = [i for i in range(game.player_count) if i != self.me and game.charges[i] > 0]
        if not opponents:
            return self.me  # If no opponents are alive, target self
        return random.choice(opponents)

    def on_opponent_move(self, move: Union[str, Tuple[int, str]], move_result):
        if move.startswith('shoot_') or move in ['beer']:
            self.known_shells = self.known_shells[1:]
    
    def on_own_move(self, last_move: str, result):
        if last_move.startswith('shoot_'):
            self.known_shells = self.known_shells[1:] if len(self.known_shells) > 0 else []
        elif last_move == 'magnifying_glass':
            self.known_shells[0] = result
        elif last_move == 'burner_phone' and result != None:
            self.known_shells[result[0]] = result[1]
        elif last_move == 'inverter' and self.known_shells[0] != None:
            self.known_shells[0] = not self.known_shells[0]
        

    def on_reload(self, game: 'BuckshotRoulette'):
        """
        Reset internal state when the game is reloaded or a new round starts.

        Args:
            game (BuckshotRoulette): The new game state.
        """
        self.last_shell = None
        self.known_shells = [None] * game.total
        self.last_shell = None  # Reset last shell information
        
class Random(AbstractEngine):
    def __init__(self, playing_as):
        self.me = playing_as
        pass
    
    def choice(self, board: BuckshotRoulette):
        ad_target, move = random.choice(board.moves())
        return move, None if ad_target == self.me else ad_target

    def on_own_move(self, last_move, res):
        pass
    
    def on_opponent_move(self, last_move, res):
        pass
    
    def on_reload(self, board: BuckshotRoulette):
        pass
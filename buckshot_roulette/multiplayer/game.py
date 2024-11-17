from enum import Enum
import random
from dataclasses import dataclass, astuple
import copy
from typing import Literal
from collections import Counter
@dataclass(init=True)
class Items():
    saw: int = 0
    magnifying_glass: int = 0
    jammer: int = 0
    cigarettes: int = 0
    beer: int = 0
    burner_phone: int = 0
    adrenaline: int = 0
    inverter: int = 0
    remote: int = 0
    
    def item_count(self):
        return self.jammer + self.magnifying_glass + self.beer + self.saw + self.cigarettes + self.inverter + self.burner_phone + self.adrenaline + self.remote
    
    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __delitem__(self, key):
        self.__setattr__(key, 0)
    
    def __iter__(self):
        return ItemIterable(self)

    def __str__(self):
        items = ', '.join(f'{key}={self[key]}' for key in self if self[key] > 0)
        return f"Items({items})"

    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        if not isinstance(other, Items):
            raise ValueError("Can only add Items objects.")
        return Items(**{key: self[key] + other[key] for key in self})
    
    def __iadd__(self, other):
        if not isinstance(other, Items):
            raise ValueError("Can only add Items objects.")
        for key in self:
            self[key] += other[key]
        return self

    def __mul__(self, factor):
        if not isinstance(factor, (int, float)):
            raise ValueError("Can only multiply by a scalar (int or float).")
        return Items(**{key: int(self[key] * factor) for key in self})

    def __imul__(self, factor):
        if not isinstance(factor, (int, float)):
            raise ValueError("Can only multiply by a scalar (int or float).")
        for key in self:
            self[key] = int(self[key] * factor)
        return self
class ItemIterable():
    def __init__(self, data: Items):
        self.data: Items = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(BuckshotRoulette.POSSIBLE_ITEMS):
            while self.index < len(BuckshotRoulette.POSSIBLE_ITEMS) and self.data[BuckshotRoulette.POSSIBLE_ITEMS[self.index]] < 1:
                self.index += 1                
            if self.index >= len(BuckshotRoulette.POSSIBLE_ITEMS):
                raise StopIteration
            
            item = BuckshotRoulette.POSSIBLE_ITEMS[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration

class GameStatus(Enum):
    JAMMED_0 = 0
    JAMMED_1 = 1
    JAMMED_2 = 2
    JAMMED_3 = 3
    ADRENALINE_ACTIVE = 4
    INVERTER_UNCERTAINTY = 5
    SAWED_OFF = 6

valid_sequences = {
    2: [(1, 2), (2, 1), (2, 2), (3, 2), (1, 1), (2, 3), (3, 3), (3, 1), (4, 2)],
    3: [(2, 3), (3, 2), (3, 3), (4, 3), (2, 2), (3, 4), (4, 4), (4, 2), (3, 1), (1, 1)],
    4: [(3, 4), (3, 2), (3, 3), (4, 3), (2, 2), (3, 4), (4, 4), (4, 2), (3, 1), (2, 1)],
}

class SequenceConfig:
    live: int = 0
    blank: int = 0
    item_count: int = 0
    
    def __init__(self, counts: tuple[int, int] | None = None, item_count: int | None = None, player_count: Literal[2, 3, 4] | None = None):
        if counts == None:
            if player_count == None:
                raise ValueError("Must provide the amount of players if shotgun counts are not provided!")
            counts = random.choice(valid_sequences[player_count])
        self.live = counts[0]
        self.blank = counts[1]
        
        if item_count == None:
            self.item_count = random.randint(2, 5)
        else:
            self.item_count = item_count      

    def fire(self, live: bool):
        if live:
            self.live -= 1
        else:
            self.blank -= 1

class RoundConfig:
    sequences: list[SequenceConfig | None] = [None] * 4
    start_charges: int = 4
    ITEM_CAPS = Items(saw=2, magnifying_glass=2, jammer=1, cigarettes=1, beer=8, burner_phone=8, inverter=4, adrenaline=4, remote=1)
    GLOBAL_ITEM_CAPS = Items(saw=32, magnifying_glass=32, jammer=1, cigarettes=32, beer=32, burner_phone=32, inverter=32, adrenaline=32, remote=2)
    ENABLED_ITEMS = Items(saw=1, magnifying_glass=1, jammer=1, cigarettes=1, beer=1, burner_phone=1, inverter=1, adrenaline=1, remote=1)
    
    def __init__(
        self,
        start_charges: int | None = None, 
        sequences: list[SequenceConfig | None] | None = None, 
        item_caps: Items | None = None,
        global_item_caps: Items | None = None,
        enabled_items: Items | None = None,
        player_count: int | None = None
    ):
        self.start_charges = start_charges if start_charges != None else random.randint(3, 5)
        self.sequences = sequences if sequences != None else [None] * 4
        for i in range(len(self.sequences)):
            if self.sequences[i] == None:
                if player_count == None:
                    raise ValueError("Player count may not be None if randomized sequence is passed.")
                self.sequences[i] = SequenceConfig(player_count=player_count)
        
        if item_caps != None:
            self.ITEM_CAPS = item_caps
        if global_item_caps != None:
            self.GLOBAL_ITEM_CAPS = global_item_caps
        if enabled_items != None:
            for item in enabled_items:
                if enabled_items[item] > 1:
                    enabled_items[item] = 1
                elif enabled_items[item] == 0:
                    self.GLOBAL_ITEM_CAPS[item] = 0
            self.ENABLED_ITEMS = enabled_items

class GameConfig:
    rounds = list[RoundConfig]    
    
    def __init__(self, rounds: list[RoundConfig]):
        self.rounds = rounds

class BuckshotGame:
    def __init__(self, players: list, config: GameConfig = None):
        from buckshot_roulette.multiplayer.ai import AbstractEngine
        self.player_count = len(players)
        self.players: list[AbstractEngine] = players
        
        if config == None:
            self.config = [RoundConfig(player_count=self.player_count) for _ in range(self.player_count)]
        else:
            self.config = config
        
        self.round_idx = 0
    
    def play_round(self):
        if self.config.rounds[self.round_idx] == None:
            self.config.rounds[self.round_idx] = RoundConfig(player_count=self.player_count)
        game: BuckshotRoulette = BuckshotRoulette(self.config.rounds[self.round_idx], self.player_count)
        shotgun = ([True] * game.live) + ([False] * (game.total - game.live))
        random.shuffle(shotgun)
        while game.winner() == None:
            player = self.players[game.current_turn]
            player_idx = game.current_turn
            
            move, ad_target = player.choice(game)
            res_private, res_public, new_shotgun = game.make_move(move, shotgun, ad_target)
            #print(f"[{player_idx}]: {move} -- {res_private} / {res_public} -- {new_shotgun}")
            
            # Update Shotgun- if length is 0 generate a new sequence
            if len(new_shotgun) == 0:
                shotgun = ([True] * game.live) + ([False] * (game.total - game.live))
                random.shuffle(shotgun)
                for i in range(self.player_count):
                    self.players[i].on_reload(game)
            else:
                shotgun = new_shotgun
            
            # Update all players as to the game state
            for i in range(self.player_count):
                if i == player_idx:
                    player.on_own_move(move, res_private)
                else:
                    self.players[i].on_opponent_move(move, res_public)
        victor = game.winner()
        self.round_idx += 1
        return victor

    def play(self):
        winners = {}
        while self.round_idx < len(self.config.rounds):
            winner = self.play_round()
            if winner in winners:
                winners[winner] += 1
            else:
                winners[winner] = 1

        # Find all winners that have the highest occurrence count
        modes = [winner for winner, count in winners.items() if count == max(winners.values())]
        
        # Select a winner of the overall game
        if len(modes) == 1:
            return modes[0]
        else:
            # This branch occurs in the unlikely event that no player has gotten a plurality of wins
            # Strictly speaking ties are probably broken based on time taken
            # But animation times aren't implemented and realistically this is a very unlikely scenario
            return random.choice(modes)      

class BuckshotRoulette:
    POSSIBLE_ITEMS = ['saw', 'magnifying_glass', 'jammer', 'cigarettes', 'beer', 'burner_phone', 'adrenaline', 'inverter', 'remote']
    
    def __init__(self, config: RoundConfig, player_count: int = 4):
        self.config = config        
        
        self.player_count = player_count
        
        self.charges = [self.config.start_charges] * player_count
        # TODO: Maybe randomize? Not clear how games start
        self.current_turn = 0
        # Can be flipped by remote
        self.turn_inc = 1
        
        self.items: list[Items] = [Items() for _ in range(player_count)]
        
        self.statuses: set[GameStatus] = set()
        
        # Will be incremented to 0 in first call of reload()
        self.sequence_idx = -1
        self.next_sequence()
    
    # Key is player count, Value is a list of tuples in format (live, blank)
    # From mp_main.tscn    
    
    def next_sequence(self, drop_items = True):
        self.statuses = set()
        self.sequence_idx = (self.sequence_idx + 1) % len(self.config.sequences)
        new_sequence = self.config.sequences[self.sequence_idx]
        self.total = new_sequence.live + new_sequence.blank
        self.live = new_sequence.live
        if drop_items:
            self.give_items(new_sequence.item_count)
    
    def give_items(self, item_count):
        global_count = Items()
        for player in self.items:
            for i in self.POSSIBLE_ITEMS:
                global_count[i] += player[i]
        for player in self.items:
            if player.item_count() == 8:
                # unfortunate.
                break
            choices = [i for i in self.POSSIBLE_ITEMS if player[i] < self.config.ITEM_CAPS[i] and global_count[i] < self.config.GLOBAL_ITEM_CAPS[i]]
            
            # Patch 1.2.1
            # TODO: Double check behavior with source code when someone rips it
            if self.config.start_charges <= 2 and 'saw' in choices:
                choices.remove('saw')
             
            if len(choices) > 0:
                items = random.choices(choices, k=min(item_count, 8 - player.item_count()))
                for item in items:
                    player[item] += 1
    
    def switch_turn(self):
        # Check for a winner- if so this doesn't matter
        winner = self.winner()
        if winner != None:
            return winner
        n = self.current_turn
        # We limit by player count to prevent infinite looping
        for _ in range(self.player_count * 2):
            n = (n + self.turn_inc) % self.player_count
            if GameStatus(n) in self.statuses:
                # Player jammed
                self.statuses.remove(GameStatus(n))
                continue
            if self.charges[n] > 0:
                self.current_turn = n
                return n
        raise ValueError("No valid turns available.")
    
    def offset_to_idx(self, offset):
        return (self.current_turn + offset) % self.player_count
    def idx_to_offset(self, idx):
        return (idx - self.current_turn) % self.player_count
    
    def living_players(self, as_offset=False):
        indices = [i for i in range(self.player_count) if self.charges[i] > 0]
        if as_offset:
            return [self.idx_to_offset(i) for i in indices]
        return indices
    
    def winner(self):
        players = self.living_players()
        if len(players) == 1:
            return players[0]
        return None
    
    def moves(self):
        if GameStatus.ADRENALINE_ACTIVE in self.statuses:
            moves = [] # Player MUST pick an opponent's item
            opponents = self.living_players()
            opponents.remove(self.current_turn)
            for player in opponents:
                if player == self.current_turn:
                    continue
                for item in self.items[player]:
                    if item == 'adrenaline':
                        continue
                    if item == 'jammer':
                        moves.extend([(player, f"{item}_{self.idx_to_offset(i)}") for i in opponents])
                    else:
                        moves.append((player, item))
        else:
            moves = []
            # Player may shoot any one of the currently living players
            players = self.living_players(as_offset=True)
            for player in players:
                moves.append((0, f'shoot_{player}'))
            # Item Uses
            items = self.items[self.current_turn]
            for item in items:
                if item == 'jammer':
                    moves.extend([(player, f"{item}_{i}") for i in players if i != 0])
                else:
                    moves.append((player, item))
                
        if len(moves) == 0:
            # Only possible if the previous move is adrenaline, and there are no valid items to take
            # Unfortunate.
            self.statuses.remove(GameStatus.ADRENALINE_ACTIVE)
            return self.moves()            
        return moves
    
    def make_move(
        self, 
        move: Literal['shoot_0', 'shoot_1', 'shoot_2', 'shoot_3', 'saw', 'magnifying_glass', 'jammer_1', 'jammer_2', 'jammer_3', 'cigarettes', 'beer', 'burner_phone', 'adrenaline', 'inverter', 'remote'], 
        shotgun: list[bool],
        adrenaline_target: int | None = None,
        allow_reload: bool = True
    ):
        if GameStatus.ADRENALINE_ACTIVE in self.statuses:
            if adrenaline_target == None:
                raise ValueError('Must specify which player items are being taken from if adrenaline is active.')
            target = adrenaline_target
            items = self.items[target]
            self.statuses.remove(GameStatus.ADRENALINE_ACTIVE)
        else:
            items = self.items[self.current_turn]

        out_val = None, None, shotgun

        if move.startswith('shoot'):
            idx = int(move.split('_')[1])
            is_live = shotgun[0]
            shotgun = shotgun[1:]
            target = self.offset_to_idx(idx)
            damage = 1 if is_live else 0
            self.total -= 1
            if is_live:
                self.live -= 1
            if GameStatus.SAWED_OFF in self.statuses:
                self.statuses.remove(GameStatus.SAWED_OFF)
                damage *= 2
            self.charges[target] = max(0, self.charges[target] - damage)
            if target == self.current_turn:
                if is_live:
                    self.switch_turn()
                    out_val = -damage, -damage, shotgun
                else:
                    out_val = 0, 0, shotgun
            else:
                self.switch_turn()
                out_val = damage, damage, shotgun
        elif move.startswith('jammer'):
            items.jammer -= 1
            idx = int(move.split('_')[1])
            target = self.offset_to_idx(idx)
            match target:
                case 0:
                    self.statuses.add(GameStatus.JAMMED_0)
                case 1:
                    self.statuses.add(GameStatus.JAMMED_1)
                case 2:
                    self.statuses.add(GameStatus.JAMMED_2)
                case 3:
                    self.statuses.add(GameStatus.JAMMED_3)
            out_val = target, target, shotgun        
        else:
            match move:
                case 'saw':
                    items.saw -= 1
                    self.statuses.add(GameStatus.SAWED_OFF)
                case 'magnifying_glass':
                    items.magnifying_glass -= 1
                    out_val = shotgun[0], None, shotgun
                case 'cigarettes':
                    items.cigarettes -= 1
                    self.charges[self.current_turn] = min(self.charges[self.current_turn]+1, self.config.start_charges)
                case 'beer':
                    items.beer -= 1
                    if len(shotgun) > 1:
                        val = shotgun[0]
                        shotgun = shotgun[1:]
                        out_val = val, val, shotgun
                    else:
                        shotgun = []
                        out_val = None, None, shotgun
                case 'burner_phone':
                    items.burner_phone -= 1
                    if len(shotgun) > 2:
                        idx = random.randint(2, len(shotgun)-1)
                        out_val = (idx, shotgun[idx]), None, shotgun
                case 'adrenaline':
                    items.adrenaline -= 1
                    self.statuses.add(GameStatus.ADRENALINE_ACTIVE)
                case 'inverter':
                    items.inverter -= 1
                    shotgun[0] = not shotgun[0]
                    self.statuses.add(GameStatus.INVERTER_UNCERTAINTY)
                case 'remote':
                    items.remote -= 1
                    self.turn_inc *= -1
                    out_val = self.turn_inc, self.turn_inc, shotgun
        if allow_reload and len(shotgun) == 0:
            self.next_sequence()
        return out_val
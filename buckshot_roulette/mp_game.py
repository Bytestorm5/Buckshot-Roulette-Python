from enum import Enum
import random
from dataclasses import dataclass, astuple
import copy
from typing import Literal
from buckshot_roulette.ai import Dealer
@dataclass(init=True)
class Items():
    saw: int = 0
    magnifying_glass: int = 0
    jammer: int = 0
    cigarettes: int = 0
    beer: int = 0
    burner_phone: int = 0
    meds: int = 0
    adrenaline: int = 0
    inverter: int = 0
    remote: int = 0
    
    def item_count(self):
        return self.jammer + self.magnifying_glass + self.beer + self.saw + self.cigarettes + self.inverter + self.burner_phone + self.meds + self.adrenaline + self.remote
    
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
    JAMMED_0 = 0,
    JAMMED_1 = 1,
    JAMMED_2 = 2,
    JAMMED_3 = 3,
    ADRENALINE_ACTIVE = 4,
    INVERTER_UNCERTAINTY = 5,
    SAWED_OFF = 6

class BuckshotRoulette:
    POSSIBLE_ITEMS = ['saw', 'magnifying_glass', 'jammer', 'cigarettes', 'beer', 'burner_phone', 'meds', 'adrenaline', 'inverter', 'remote']
    ITEM_CAPS = Items(saw=2, magnifying_glass=2, jammer=1, cigarettes=1, beer=8, burner_phone=8, meds=0, inverter=4, adrenaline=4, remote=1)
    GLOBAL_ITEM_CAPS = Items(saw=64, magnifying_glass=64, jammer=1, cigarettes=64, beer=64, burner_phone=64, meds=0, inverter=64, adrenaline=64, remote=2)
    def __init__(self, player_count: int = 4, start_player = 0, charge_count = None):
        self.max_charges = charge_count if charge_count else random.randint(3, 5)
        
        self.player_count = player_count
        
        self.charges = [self.max_charges] * player_count
        self.starter = start_player
        self.current_turn = start_player
        # Can be flipped by remote
        self.turn_inc = 1
        
        self.items: list[Items] = [Items() for _ in range(player_count)]
        
        self.statuses: set[GameStatus] = set()
        
        self.chamber_public = None
        self.give_items(random.randint(2, 5))
    
    # Key is player count, Value is a list of tuples in format (live, blank)
    # From mp_main.tscn
    valid_sequences = {
        2: [(1, 2), (2, 1), (2, 2), (3, 2), (1, 1), (2, 3), (3, 3), (3, 1), (4, 2)],
        3: [(2, 3), (3, 2), (3, 3), (4, 3), (2, 2), (3, 4), (4, 4), (4, 2), (3, 1), (1, 1)],
        4: [(3, 4), (3, 2), (3, 3), (4, 3), (2, 2), (3, 4), (4, 4), (4, 2), (3, 1), (2, 1)],
    }
    
    def reload(self, drop_items = True):
        self.statuses = set()
        new_arrangement = random.choice(self.valid_sequences[self.player_count])
        self.total = sum(new_arrangement)
        self.live = new_arrangement[0]
        if drop_items:
            self.give_items(random.randint(3, 5))
    
    def give_items(self, item_count):
        global_count = Items()
        for player in self.items:
            for i in self.POSSIBLE_ITEMS:
                global_count[i] += player[i]
        for player in self.items:
            if player.item_count() == 8:
                # unfortunate.
                break
            choices = [i for i in self.POSSIBLE_ITEMS if player[i] < self.ITEM_CAPS[i] and global_count[i] < self.GLOBAL_ITEM_CAPS[i]]
            
            # Patch 1.2.1
            # TODO: Double check behavior with source code when someone rips it
            if self.max_charges <= 2 and 'saw' in choices:
                choices.remove('saw')
                
            items = random.choices(choices, k=min(item_count, 8 - player.item_count()))
            for item in items:
                player[item] += 1
    
    def switch_turn(self):
        n = self.current_turn
        # We limit by player count to prevent infinite looping
        for _ in range(self.player_count):
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
            players = self.living_players(as_offset=True)
            for player in players:
                moves.append((0, f'shoot_{player}'))
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
        move: Literal['shoot_0', 'shoot_1', 'shoot_2', 'shoot_3', 'saw', 'magnifying_glass', 'jammer_1', 'jammer_2', 'jammer_3', 'cigarettes', 'beer', 'burner_phone', 'meds', 'adrenaline', 'inverter', 'remote'], 
        shotgun: list[bool],
        adrenaline_target: int | None = None
    ):
        if GameStatus.ADRENALINE_ACTIVE in self.statuses:
            if adrenaline_target == None:
                raise ValueError('Must specify which player items are being taken from if adrenaline is active.')
            target = adrenaline_target + self.current_turn
            items = self.items[target]
            self.statuses.remove(GameStatus.ADRENALINE_ACTIVE)
        else:
            items = self.items[self.current_turn]
        
        if move.startswith('shoot'):
            idx = int(move.split('_')[0])
            target = self.current_turn + idx
            damage = 1
            if GameStatus.SAWED_OFF in self.statuses:
                self.statuses.remove(GameStatus.SAWED_OFF)
                damage = 2
            self.charges = max(0, self.charges[target] - damage)
            return -damage if target == self.current_turn else damage
        
        if move.startswith('jammer'):
            items.jammer -= 1
            idx = int(move.split('_')[0])
            target = self.current_turn + idx
            match target:
                case 1:
                    self.statuses.add(GameStatus.JAMMED_1)
                case 2:
                    self.statuses.add(GameStatus.JAMMED_2)
                case 3:
                    self.statuses.add(GameStatus.JAMMED_3)
        out_val = None
        match move:
            case 'saw':
                items.saw -= 1
                self.statuses.add(GameStatus.SAWED_OFF)
                pass
            case 'magnifying_glass':
                items.magnifying_glass -= 1
                out_val = shotgun[0]
            case 'cigarettes':
                items.cigarettes -= 1
                self.charges[self.current_turn] = min(self.charges[self.current_turn]+1, self.max_charges)
            case 'beer':
                items.beer -= 1
                if len(shotgun) > 1:
                    val = shotgun[0]
                    shotgun = shotgun[1:]
                    out_val = val
                else:
                    shotgun = []
            case 'burner_phone':
                items.burner_phone -= 1
                if len(shotgun) > 2:
                    idx = random.randint(2, len(shotgun)-1)
                    out_val = (idx, shotgun[idx])
            case 'meds':
                raise ValueError('Expired Meds are not valid for Multiplayer!')
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
        return out_val
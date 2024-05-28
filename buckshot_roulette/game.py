import random
from dataclasses import dataclass, astuple
import copy

@dataclass(init=True)
class Items():
    handcuffs: int = 0
    magnifying_glass: int = 0
    beer: int = 0
    saw: int = 0
    cigarettes: int = 0
    inverter: int = 0
    burner_phone: int = 0
    meds: int = 0
    adrenaline: int = 0
    
    def item_count(self):
        return self.handcuffs + self.magnifying_glass + self.beer + self.saw + self.cigarettes + self.inverter + self.burner_phone + self.meds + self.adrenaline
    
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

class BuckshotRoulette:
    POSSIBLE_ITEMS = ['handcuffs', 'magnifying_glass', 'beer', 'cigarettes', 'saw', 'inverter', 'burner_phone', 'meds', 'adrenaline']
    ITEM_CAPS = Items(handcuffs=1, magnifying_glass=3, beer=2, cigarettes=1, saw=3, inverter=8, burner_phone=1, meds=1, adrenaline=2)
    def __init__(self, charge_count = None, total_rounds = None, live_rounds = None):
        self.max_charges = charge_count if charge_count else random.randint(2, 4)
        self.charges = [self.max_charges, self.max_charges]
        self.current_turn = 0
        
        total = total_rounds if total_rounds else random.randint(2, 8)
        live = total // 2 if live_rounds == None else live_rounds
        if live > total:
            raise ValueError("Live Rounds must be less than Total Rounds")
        
        self._shotgun = ([True] * live) + ([False * (total - live)])
        random.shuffle(self._shotgun)
        
        self.items: list[Items] = [
            Items(),
            Items()
        ]
        #self.p2_items: Items = {'handcuffs': 0, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0}
        
        self._active_items: Items = Items()
        self._skip_next = False
        
        self.chamber_public = None
        self.give_items(random.randint(2, 5))       

    def new_rounds(self, drop_items = True):
        total = random.randint(2, 8)
        live = total // 2
        self._shotgun = ([True] * live) + ([False] * (total - live))
        random.shuffle(self._shotgun)
        if drop_items:
            self.give_items(random.randint(2, 5))
    
    def give_items(self, item_count):
        for player in self.items:
            if player.item_count() == 8:
                # unfortunate.
                break
            choices = [i for i in self.POSSIBLE_ITEMS if player[i] < self.ITEM_CAPS[i]]
            
            # Patch 1.2.1
            # TODO: Double check behavior with source code when someone rips it
            if self.max_charges <= 2 and 'saw' in choices:
                choices.remove('saw')
                
            items = random.choices(choices, k=min(item_count, 8 - player.item_count()))
            for item in items:
                player[item] += 1
    
    def shotgun_info(self):
        live = sum([1 if x else 0 for x in self._shotgun])
        return live, len(self._shotgun)
    
    def winner(self) -> int | None:
        # Ties are impossible so we don't account for that
        if self.charges[0] < 1:
            return 1
        elif self.charges[1] < 1:
            return 0
        else:
            return None        
        
    def fire(self, at_opponent=True) -> None:
        target = (self.current_turn + at_opponent) % 2
        is_hit = self._shotgun[0]
        self._shotgun = self._shotgun[1:]
        self.chamber_public = None
        
        def switch():
            self._active_items.saw = 0
            if self._active_items.handcuffs > 0.5:
                if not at_opponent and not is_hit:
                    return
                self._active_items.handcuffs -= 0.5
                self._skip_next = True
            
            if self._skip_next:
                self._skip_next = False
            else:
                self.switch_turn()
                        
        if is_hit:
            damage = 1
            if self._active_items.saw > 0:
                self._active_items.saw = 0
                damage = 2
            self.charges[target] -= damage
            switch()
            return damage
        elif at_opponent: # Missed against opponent
            switch()
            return 0
        else: # Shot at self and missed
            return 0
    
    def legal_items(self) -> list[str]:
        """All items that could be used in the current board state, regardless of whether the player currently has them
        """
        out_arr = []
        for item in self.POSSIBLE_ITEMS:
            if self._active_items[item] == 0:
                out_arr.append(item)
        return out_arr
    
    def moves(self):
        if self._active_items.adrenaline > 0:
            moves = [] # Player MUST pick an opponent's item
            items = self.items[self.opponent()]
        else:
            moves = ['op', 'self']
            items = self.items[self.current_turn]
        
        for item in self.POSSIBLE_ITEMS:            
            if items[item] > 0 and self._active_items[item] == 0:
                moves.append(item)
                
        if len(moves) == 0:
            # Only possible if the previous move is adrenaline, and there are no valid items to take
            # Unfortunate.
            self._active_items.adrenaline = 0
            return self.moves()            
        return moves

    def make_move(self, move, load_new = True):
        out_val = None
        if self._active_items.adrenaline > 0:
            items = self.items[self.opponent()]
            self._active_items.adrenaline = 0
        else:
            items = self.items[self.current_turn]
        match move:
            case 'op':
                out_val = self.fire(at_opponent=True)
            case 'self':
                out_val = -self.fire(at_opponent=False)
            case 'handcuffs':
                items.handcuffs -= 1
                self._active_items.handcuffs += 1
                self._skip_next = True
            case 'magnifying_glass':
                items.magnifying_glass -= 1
                out_val = self._shotgun[0]
                self.chamber_public = self._shotgun[0]
            case 'beer':
                items.beer -= 1
                if len(self._shotgun) > 1:
                    val = self._shotgun[0]
                    self._shotgun = self._shotgun[1:]
                    out_val = val
                else:
                    self._shotgun = []
            case 'cigarettes':
                items.cigarettes -= 1
                self.charges[self.current_turn] = min(self.charges[self.current_turn]+1, self.max_charges)
            case 'saw':
                items.saw -= 1
                self._active_items.saw += 1
            case 'inverter':
                items.inverter -= 1
                self._shotgun[0] = not self._shotgun[0]
            case 'burner_phone':
                items.burner_phone -= 1
                if len(self._shotgun) > 1:
                    idx = random.randint(1, len(self._shotgun)-1)
                    out_val = (idx, self._shotgun[idx])
            case 'meds':
                items.meds -= 1
                if random.random() > 0.5:
                    self.charges[self.current_turn] = min(self.charges[self.current_turn] + 2, self.max_charges)
                else:
                    self.charges[self.current_turn] -= 1
            case 'adrenaline':
                items.adrenaline -= 1
                self._active_items.adrenaline += 1
                    
        
        if load_new and len(self._shotgun) == 0:
            self.current_turn = 0
            self.new_rounds()
            return out_val
        
        return out_val
    
    def live_round(self):
        return self._shotgun[0]
    
    def switch_turn(self):
        #self._active_items = {'handcuffs': 0, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0}
        self.current_turn = 1 if self.current_turn == 0 else 0
        
    def opponent(self):
        return 1 if self.current_turn == 0 else 0

    def copy(self):
        new_board = BuckshotRoulette(charge_count=0)  # Temporary charge count; will be overwritten
        new_board.charges = self.charges[:]
        new_board.current_turn = self.current_turn
        new_board._shotgun = self._shotgun[:]
        new_board.items = copy.deepcopy(self.items)        
        new_board._active_items = copy.deepcopy(self._active_items)
        return new_board
    
    def __eq__(self, other):
        if not isinstance(other, BuckshotRoulette):
            return NotImplemented
        
        # Comparing all relevant attributes for equality
        return (self.max_charges == other.max_charges and
                self.charges == other.charges and
                self.current_turn == other.current_turn and
                self._shotgun == other._shotgun and
                self.items == other.items and
                self._active_items == other._active_items and
                self._skip_next == other._skip_next and
                self.chamber_public == other.chamber_public)

    def __hash__(self):
        # Creating a hash based on a tuple of immutable representations of relevant attributes
        return hash((self.max_charges, 
                     tuple(self.charges), 
                     self.current_turn, 
                     tuple(self._shotgun), 
                     tuple(astuple(player) for player in self.items), 
                     astuple(self._active_items), 
                     self._skip_next, 
                     self.chamber_public))
    def to_json(self):
        return {
            "max_charges": self.max_charges,
            "charges": self.charges,
            "current_turn": self.current_turn,
            "shotgun": self._shotgun,
            "items": self.items,
            "active_items": self._active_items,
            "skip_next": self._skip_next,
            "chamber_public": self.chamber_public
        }
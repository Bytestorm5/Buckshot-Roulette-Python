import random
from typing import TypedDict
import itertools
import math
import copy

class Items(TypedDict):
    handcuffs: int
    magnifying_glass: int
    beer: int
    saw: int
    cigarettes: int
    
class Board:
    POSSIBLE_ITEMS = ['handcuffs', 'magnifying_glass', 'beer', 'cigarettes', 'saw']
    def __init__(self, charge_count, total_rounds = None, live_rounds = None):
        self.max_charges = charge_count
        self.charges = [charge_count, charge_count]
        self.current_turn = 0
        
        total = total_rounds if total_rounds else random.randint(2, 8)
        live = total // 2 if live_rounds == None else live_rounds
        if live > total:
            raise ValueError("Live Rounds must be less than Total Rounds")
        
        self._shotgun = random.choice(generate_binary_numbers(live, total))
        
        self.items: list[Items] = [
            {'handcuffs': 0, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0},
            {'handcuffs': 0, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0}
        ]
        #self.p2_items: Items = {'handcuffs': 0, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0}
        
        self._active_items: Items = {'handcuffs': 0, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0}
        self._skip_next = False
        
        self.chamber_public = None

    def new_rounds(self, drop_items = True):
        total = random.randint(2, 8)
        live = random.randint(1, total-1)
        self._shotgun = random.choice(generate_binary_numbers(live, total))
        if drop_items:
            self.give_items(random.randint(1, 4))
            
    
    def give_items(self, item_count):
        for player in self.items:
            for _ in range(item_count):
                if sum(player.values()) == 8:
                    break
                item = random.choice(self.POSSIBLE_ITEMS)
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
            self._active_items['saw'] = 0
            if self._active_items['handcuffs'] > 0.5:
                if not at_opponent and not is_hit:
                    return
                self._active_items['handcuffs'] -= 0.5
                self._skip_next = True
            
            if self._skip_next:
                self._skip_next = False
            else:
                self.switch_turn()
                        
        if is_hit:
            damage = 1
            if self._active_items['saw'] > 0:
                self._active_items['saw'] = 0
                damage = 2
            self.charges[target] -= damage
            switch()
            return damage
        elif at_opponent: # Missed against opponent
            switch()
            return 0
        else: # Shot at self and missed
            return 0
    
    def moves(self):
        items = self.items[self.current_turn]
        moves = ['op', 'self']
        for item in self.POSSIBLE_ITEMS:            
            if items[item] > 0 and self._active_items[item] == 0:
                moves.append(item)
        return moves

    def make_move(self, move, load_new = True):
        out_val = None
        items = self.items[self.current_turn]
        match move:
            case 'op':
                out_val = self.fire(at_opponent=True)
            case 'self':
                out_val = -self.fire(at_opponent=False)
            case 'handcuffs':
                items['handcuffs'] -= 1
                self._active_items['handcuffs'] += 1
                self._skip_next = True
            case 'magnifying_glass':
                items['magnifying_glass'] -= 1
                out_val = self._shotgun[0]
                self.chamber_public = self._shotgun[0]
            case 'beer':
                items['beer'] -= 1
                if len(self._shotgun) > 1:
                    val = self._shotgun[0]
                    self._shotgun = self._shotgun[1:]
                    out_val = val
                else:
                    self._shotgun = []
            case 'cigarettes':
                items['cigarettes'] -= 1
                self.charges[self.current_turn] = min(self.charges[self.current_turn]+1, self.max_charges)
            case 'saw':
                items['saw'] -= 1
                self._active_items['saw'] += 1
        
        if load_new and len(self._shotgun) == 0:
            self.current_turn = 0
            self.new_rounds()
        
        return out_val
    
    def live_round(self):
        return self._shotgun[0]
    
    def switch_turn(self):
        #self._active_items = {'handcuffs': 0, 'magnifying_glass': 0, 'beer': 0, 'cigarettes': 0, 'saw': 0}
        self.current_turn = 1 if self.current_turn == 0 else 0
        
    def opponent(self):
        return 1 if self.current_turn == 0 else 0

    def copy(self):
        new_board = Board(charge_count=0)  # Temporary charge count; will be overwritten
        new_board.charges = self.charges[:]
        new_board.current_turn = self.current_turn
        new_board._shotgun = self._shotgun[:]  # Assuming shotgun is a list
        new_board.items = copy.deepcopy(self.items)        
        new_board._active_items = copy.deepcopy(self._active_items)
        return new_board
    
    def __eq__(self, other):
        if not isinstance(other, Board):
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
                     tuple(tuple(sorted(player.items())) for player in self.items), 
                     tuple(sorted(self._active_items.items())), 
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
def generate_binary_numbers(X, N):
    if X > N or X < 0 or N < 0:
        raise ValueError("Invalid inputs. Ensure 0 <= X <= N.")

    binary_numbers = []

    # Generate all combinations of indices where 1s can be placed
    for indices in itertools.combinations(range(N), X):
        # Create a binary number with 0s and set 1s at the specific indices
        binary_number = [False] * N
        for index in indices:
            binary_number[index] = True
        binary_numbers.append(binary_number)

    return binary_numbers

def binomial_coefficient(X, N):
    """Calculate the binomial coefficient 'N choose X'."""
    if X < 0 or N < 0 or X > N:
        raise ValueError("Invalid inputs. Ensure 0 <= X <= N.")
    
    return math.factorial(N) // ( math.factorial(X) *  math.factorial(N - X))
  
if __name__ == "__main__":
    board = Board(5)
    board.__hash__()
    
    live = sum([1 if x else 0 for x in board._shotgun])
    while board.winner() == None:
        live = sum([1 if x else 0 for x in board._shotgun])
        print(f"{live} Live, {len(board._shotgun) - live} Blank.")
        if board.current_turn == 0:
            print("Charges:")
            print(board.charges)
            print()
            print("Active Items:")
            print(board._active_items)
            print()
            print("Your Items")
            print(board.p1_items)
            print()
            print("Opponent Items")
            print(board.p2_items)
            
            moves = board.moves()
            for i, move in enumerate(moves):
                print(f'[{i}]: {move}')
            
            idx = int(input("Enter Move here (0-indexed): "))
            print("Result:", board.make_move(moves[idx]))
            print("------------------------------")
        else:
            while board.current_turn == 1:
                move = random.choice(board.moves())
                print("Bot Used:", move)
                print("Result:", board.make_move(move))
                print("------------------------------")
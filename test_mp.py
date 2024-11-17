import random
from buckshot_roulette.multiplayer.game import SequenceConfig, Items, BuckshotRoulette, RoundConfig, GameConfig, BuckshotGame
from buckshot_roulette.multiplayer.ai import Random, Dealer

def generate_random_game_config(player_count: int):
    num_rounds = random.randint(1, 5)
    rounds = []
    for _ in range(num_rounds):
        # Randomly decide whether to include a RoundConfig or None
        if random.choice([True, False]):
            round_config = None
        else:
            # Randomly decide values or None for RoundConfig parameters
            start_charges = random.choice([None, random.randint(1, 5)])
            
            sequences = None
            if random.choice([True, False]):
                num_sequences = random.randint(1, 4)
                sequences = []
                for _ in range(num_sequences):
                    if random.choice([True, False]):
                        sequences.append(None)
                    else:
                        counts = None
                        if random.choice([True, False]):
                            counts = (random.randint(1, 4), random.randint(1, 4))
                        item_count = random.choice([None, random.randint(2, 5)])
                        seq_config = SequenceConfig(counts=counts, item_count=item_count, player_count=player_count)
                        sequences.append(seq_config)
                        
            item_caps = None
            if random.choice([True, False]):
                item_caps = Items(**{item: random.randint(0, 10) for item in BuckshotRoulette.POSSIBLE_ITEMS})
            
            global_item_caps = None
            if random.choice([True, False]):
                global_item_caps = Items(**{item: random.randint(0, 10) for item in BuckshotRoulette.POSSIBLE_ITEMS})
            
            enabled_items = None
            if random.choice([True, False]):
                enabled_items = Items(**{item: random.choice([0, 1]) for item in BuckshotRoulette.POSSIBLE_ITEMS})
            
            round_config = RoundConfig(
                start_charges=start_charges,
                sequences=sequences,
                item_caps=item_caps,
                global_item_caps=global_item_caps,
                enabled_items=enabled_items,
                player_count=player_count
            )
        rounds.append(round_config)
    game_config = GameConfig(rounds=rounds)
    return game_config

win_counts = [0] * 4
for _ in range(10000):
    player_count = 4
    random_game_config = generate_random_game_config(player_count)

    players = [Dealer(_) for _ in range(player_count)]
    game = BuckshotGame(players, random_game_config)
    winner = game.play()
    if winner != None:
        win_counts[winner] += 1
    if _ == 0:
        continue
    print(f"[{_} / 10000]: 0: {win_counts[0] / _:.2f}, 1: {win_counts[1] / _:.2f}, 2: {win_counts[2] / _:.2f}, 3: {win_counts[3] / _:.2f}")
print(win_counts)

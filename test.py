from buckshot_roulette.game import BuckshotRoulette, BuckshotGame
from buckshot_roulette.ai import *



if True:
    # ai vs ai
    engine0 = Random(0)
    engine1 = Dealer(1)
    game = BuckshotGame(engine0, engine1)
    p1_wins = 0
    n = 10000
    for i in range(1,n+1):
        p1_wins += game.play(charges = 4, celebrate= False, itemsused= False)
    print(f"the dealer has a {100*p1_wins / n:.2f}% win rate")

if False:
    # human vs dealer
    # ai vs ai
    engine0 = Human(0, knowledge=True)
    engine1 = Dealer(1)
    game = BuckshotGame(engine0, engine1)
    p1_wins += game.play(charges = 4, celebrate= True, itemsused= True)
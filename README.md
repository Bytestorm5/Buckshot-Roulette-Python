# Buckshot Roulette
> BETA- Things may be broken or not work as intended. If you find something wrong, make a PR or an Issue!

A Buckshot Roulette library ([PyPI](https://pypi.org/project/buckshot-roulette/)) for python, with complete game features & reasonable efficiency.

[What is Buckshot Roulette?](https://store.steampowered.com/app/2835570/Buckshot_Roulette/)

Mainly intended for use in developing engines to play the game optimally.

## Sources
Code in this library is developed to match the behaviors of the actual game, using source code decompiled by [thecatontheceiling](https://github.com/thecatontheceiling).

[Source Code for Singleplayer](https://github.com/thecatontheceiling/buckshotroulette)

[Source Code for Multiplayer](https://github.com/thecatontheceiling/buckshotroulette_multiplayer)

## Quickstart
```
pip install buckshot-roulette
```
```python
from buckshot_roulette import BuckshotRoulette

board = BuckshotRoulette(charge_count=4)
```

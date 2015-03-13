"""
RL Assignment - Easy21
@author: William Ferreira

Constants.
"""
# Game reward values
WIN = 1
DRAW = 0
LOSE = -1

# Points limits
MIN_POINTS = 1
MAX_POINTS = 21

# Player game actions
HIT = 'hit'
STICK = 'stick'
ACTIONS = (HIT, STICK)

NUMBER_OF_CARDS = 10

RED_PROBABILITY = 1.0/3.0
BLACK_PROBABILITY = 1.0 - RED_PROBABILITY

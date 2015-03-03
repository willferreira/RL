"""
RL Assignment - Easy21
@author: William Ferreira

1 Implementation of Easy21

You should write an environment that implements the game Easy21. Specifically,
write a function, named step, which takes as input a state s (dealer’s first
card 1–10 and the player’s sum 1–21), and an action a (hit or stick), and returns a
sample of the next state s′ (which may be terminal if the game is finished) and
reward r. We will be using this environment for model-free reinforcement learning,
and you should not explicitly represent the transition matrix for the MDP. There
is no discounting (γ = 1). You should treat the dealer’s moves as part of the
environment, i.e. calling step with a stick action will play out the dealer’s cards
and return the final reward and terminal state.

                                                                          10 marks
"""
import itertools as it
import logging

from scipy.stats import rv_discrete

# Create the deck.
# a card is a tuple (x, y) where:
#   x is either 'R' or 'B' for a red or black suit card
#   y is the card number: 1-10
BLACK = 'B'
RED = 'R'

_SUITS = (RED, BLACK)
_NO_CARDS_IN_SUIT = 10
_deck = list(it.product(_SUITS, range(1, _NO_CARDS_IN_SUIT+1)))

_logger = logging.getLogger(__name__)


# Create a discrete probability distribution over the deck
# where:
#   distribution of card number is U(1, 10)
#   the probability of a red card is 1/3 and a black card is 2/3
_suit_probabilities = {RED: 1.0/3.0}
_suit_probabilities[BLACK] = 1.0 - _suit_probabilities[RED]


def _get_card_probabilities(suit):
    # generate the card probabilities for the given suit
    return [_suit_probabilities[suit]/_NO_CARDS_IN_SUIT]*_NO_CARDS_IN_SUIT

_deck_probabilities = it.chain((_get_card_probabilities(suit) for suit in _SUITS))
_easy21_deck_distribution = rv_discrete(values=(range(len(_deck)), list(_deck_probabilities)),
                                        name='Easy21 Deck Distribution')
_uniform_1_10_distribution = rv_discrete(values=(range(1, _NO_CARDS_IN_SUIT+1),
                                                 [1.0/_NO_CARDS_IN_SUIT]*_NO_CARDS_IN_SUIT),
                                         name='U(1, 10) Distribution')


def draw_from_deck_with_replacement(initial=False):
    """
    Draw a card from the Easy21 deck distribution, with replacement.

    :param initial: If True then draw a black card uniformly, otherwise
                    draw a card from the full deck, always with replacement.
    :return: a card value drawn randomly with replacement from the deck
    """
    if initial:
        draw = _uniform_1_10_distribution.rvs(size=1)[0]
    else:
        card, value = _deck[_easy21_deck_distribution.rvs(size=1)]
        draw = value * (1 if card == 'B' else -1)
        _logger.log(logging.DEBUG, 'Draw: {0:s}'.format(str((card, draw))))
    return draw


# Game reward values
WIN = 1
DRAW = 0
LOSE = -1

# Player game actions
HIT = 'hit'
STICK = 'stick'
ACTIONS = (HIT, STICK)


def is_busted(score):
    """
    Returns true if the score is busted, False otherwise.
    :param score: the score
    :return: Returns true if the score is busted, False otherwise.
    """
    return score < 1 or score > 21


def is_terminal(s):
    """
    Returns True if s is a terminal state, False otherwise.
    :param s: A tuple (x, y) representing the state where:
                x - the dealer's first card, 1-10
                y - the player's sum: 1-21
    :return: True if s is a terminal state, i.e. either
             player or dealer is bust, and False otherwise.
    """
    return is_busted(s[0]) or is_busted(s[1])


def step(s, a):
    """
    Function which takes as input a state s (dealer’s first card 1–10 and the
    player’s sum 1–21), and an action a (hit or stick), and returns a sample of the next
    state s′ (which may be terminal if the game is finished) and reward r.

    :param s: A tuple (x, y) representing the state where:
                x - the dealer's first card, 1-10
                y - the player's sum: 1-21
    :param a: The action: 'stick' or 'hit'
    :return: A new sample state and the associated reward
    """
    dealer, player = s

    if a == HIT:  # player hits
        _logger.log(logging.DEBUG, 'Player draws')
        player += draw_from_deck_with_replacement()
        reward = LOSE if is_busted(player) else DRAW
    else:  # a == STICK - player sticks, dealer to play
        _logger.log(logging.DEBUG, 'Player sticks')
        while dealer < 17 and not is_busted(dealer):
            _logger.log(logging.DEBUG, 'Dealer draws')
            dealer += draw_from_deck_with_replacement()
            _logger.log(logging.DEBUG, 'Dealer new level: {0:d}'.format(dealer))
        if is_busted(dealer):
            reward = WIN
        elif dealer == player:
            reward = DRAW
        else:
            reward = WIN if player > dealer else LOSE
    return (dealer, player), reward



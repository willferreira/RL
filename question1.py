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
from scipy.stats import rv_discrete

_BLACK = 'B'
_RED = 'R'

_suits = (_RED, _BLACK)
_suit_probs = {_RED: 1.0/3.0}
_suit_probs[_BLACK] = 1.0 - _suit_probs[_RED]
_card_numbers = range(1, 11)

_card_number_dist = rv_discrete(values=(range(len(_card_numbers)), [1.0/len(_card_numbers)]*len(_card_numbers)),
                                name='Easy21 Card Number Distribution')
_suit_dist = rv_discrete(values=(range(len(_suits)), [_suit_probs[_suits[0]], _suit_probs[_suits[1]]]),
                         name='Easy21 Suit Distribution')


def draw_from_deck_with_replacement(initial=False):
    """
    Draw a card from the Easy21 deck distribution, with replacement.

    :param initial: If True then draw a black card uniformly, otherwise
                    draw a card from the full deck, always with replacement.
    :return: a card value drawn randomly with replacement from the deck
    """
    draw = _card_numbers[_card_number_dist.rvs(size=1)[0]]
    if not initial:
        suit = _suits[_suit_dist.rvs(size=1)[0]]
        draw *= (1 if suit == _BLACK else -1)
    return draw


# Game reward values
WIN = 1
DRAW = 0
LOSE = -1

# Player game actions
HIT = 'hit'
STICK = 'stick'
ACTIONS = (HIT, STICK)


def _is_busted(score):
    return score < 1 or score > 21


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
    dealer_first, player = s

    if a == HIT:  # player hits
        card = draw_from_deck_with_replacement()
        if _is_busted(player + card):
            reward = LOSE
        else:
            player += card
            reward = DRAW
    else:  # a == STICK - player sticks, dealer to play
        dealer = dealer_first
        while dealer < 17 and not _is_busted(dealer):
            dealer += draw_from_deck_with_replacement()
        if _is_busted(dealer):
            reward = WIN
        elif dealer == player:
            reward = DRAW
        else:
            reward = WIN if player > dealer else LOSE
    return (dealer_first, player), reward



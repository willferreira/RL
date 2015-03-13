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
import random as rd

from constants import *


def draw_from_deck_with_replacement(initial=False):
    """
    Draw a card from the Easy21 deck distribution, with replacement.

    :param initial: If True then draw a black card uniformly, otherwise
                    draw a card from the full deck, always with replacement.
    :return: a card value drawn randomly with replacement from the deck
    """
    draw = rd.randint(1, NUMBER_OF_CARDS)
    if not initial:
        draw *= -1 if rd.random() < RED_PROBABILITY else 1.0
    return draw


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
    def is_busted(score):
        return score < MIN_POINTS or score > MAX_POINTS

    dealer_first, player = s

    if a == HIT:  # player hits
        card = draw_from_deck_with_replacement()
        if is_busted(player + card):
            reward = LOSE
        else:
            player += card
            reward = DRAW
    else:  # a == STICK - player sticks, dealer to play
        dealer = dealer_first
        while dealer < 17 and not is_busted(dealer):
            dealer += draw_from_deck_with_replacement()
        if is_busted(dealer):
            reward = WIN
        elif dealer == player:
            reward = DRAW
        else:
            reward = WIN if player > dealer else LOSE
    return (dealer_first, player), reward



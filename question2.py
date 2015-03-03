"""
RL Assignment - Easy21
@author: William Ferreira

2 Monte-Carlo Control in Easy21

Apply Monte-Carlo control to Easy21. Initialise the value function
to zero. Use a time-varying scalar step-size of αt = 1/N(st,at) and
an ε-greedy exploration strategy with εt = N0/(N0 + N(st)), where
N0 = 100 is a constant, N(s) is the number of times that state s has
been visited, and N(s,a) is the number of times that action a has been
selected from state s. Feel free to choose an alternative value for N0,
if it helps producing better results. Plot the optimal value function
V∗ (s) = max_a Q∗ (s, a) using similar axes to the following figure taken
from Sutton and Barto’s Blackjack example.

                                                              15 marks
"""
from collections import defaultdict
import itertools as it
import logging
    
import numpy as np
from scipy.stats import rv_discrete
import pandas as pd

from question1 import (
    HIT,
    STICK,
    ACTIONS,
    step,
    is_terminal,
    draw_from_deck_with_replacement,
)

_logger = logging.getLogger(__name__)


def _generate_initial_state():
    # Returns a random initial game state, ie. dealer's show card and initial player card
    d, p = draw_from_deck_with_replacement(True), draw_from_deck_with_replacement(True)
    _logger.log(logging.DEBUG, 'Initial draw: {0:s}'.format(str((d, p))))
    return d, p


def _generate_initial_policy():
    # Generate the initial pi(s, a) distribution
    r = np.random.rand(10, 21)
    policy = pd.Panel.from_dict({HIT: pd.DataFrame(index=range(1, 11), columns=range(1, 22), data=r),
                                 STICK: pd.DataFrame(index=range(1, 11), columns=range(1, 22), data=1-r)})
    return policy


def _draw_action(s, policy):
    # Draw an action given a state and pi(s, a) policy distribution
    action_dist = rv_discrete(values=(range(0, len(policy.items)), policy.ix[:, s[0], s[1]].values))
    return policy.items[action_dist.rvs(size=1)[0]]


def generate_episode(policy):
    s = _generate_initial_state()
    episode = []
    game_ended = False
    while not game_ended:
        _logger.log(logging.DEBUG, 'State before: {0:s}'.format(str(s)))
        a = _draw_action(s, policy)
        _logger.log(logging.DEBUG, 'Action: {0:s}'.format(a))
        s1, r = step(s, a)
        episode.append((s, a, r))
        game_ended = (a == STICK) or is_terminal(s1)
        s = s1
    return episode


class Easy21MCControl(object):

    def __init__(self, T=100000, N0=100.0):
        self.T = T
        self.N0 = N0
        self.policy = _generate_initial_policy()
        self.N = defaultdict(int)
        self.Q = pd.Panel.from_dict({HIT: pd.DataFrame(index=range(1, 11), columns=range(1, 22), data=0.0),
                                     STICK: pd.DataFrame(index=range(1, 11), columns=range(1, 22), data=0.0)})
        self.alpha = defaultdict(lambda: next(it.repeat(1.0)))
        self.eta = defaultdict(lambda: next(it.repeat(N0)))
        self.V = defaultdict(float)

    def _update_Q(self, episode, reward):
        for s, a, _ in episode:
            self.Q.ix[a, s[0], s[1]] += self.alpha[(s, a)] * (reward - self.Q.ix[a, s[0], s[1]])

    def _update_alpha(self, episode):
        for s, a, _ in episode:
            self.alpha[(s, a)] = 1/self.N[(s, a)]

    def _update_eta(self, episode):
        for s, _, _ in episode:
            self.eta[s] = self.N0/(self.N0 + self.N[s])

    def _update_N(self, episode):
        for s, a, _ in episode:
            self.N[s] += 1
            self.N[(s, a)] += 1

    def run(self):
        t = 0
        while t < self.T:
            if t % 10000 == 0:
                _logger.log(logging.INFO, 't: {0:d}'.format(t))
            # Generate an episode and extract the (terminal) reward
            episode = generate_episode(self.policy)
            _, _, reward = episode[-1]

            # Update state
            self._update_Q(episode, reward)
            self._update_N(episode)
            self._update_alpha(episode)
            self._update_eta(episode)

            # Adjust policy
            for s, _, _ in episode:
                a_star = ACTIONS[np.argmax([self.Q.ix[a, s[0], s[1]] for a in ACTIONS])]
                eta = self.eta[s]
                for a in ACTIONS:
                    if a == a_star:
                        self.policy.ix[a, s[0], s[1]] = 1 - eta + eta/len(ACTIONS)
                    else:
                        self.policy.ix[a, s[0], s[1]] = eta/len(ACTIONS)
            t += 1
        self.V = self.Q.max(axis=0)


if __name__ == '__main__':
    easy21MCC = Easy21MCControl(T=100000)
    easy21MCC.run()
    print(easy21MCC.V)

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
import os
from collections import defaultdict
import itertools as it
import random as rd

from constants import *
from question1 import step, draw_from_deck_with_replacement


def generate_initial_state():
    # Returns a random initial game state, ie. dealer's show card and initial player card
    return draw_from_deck_with_replacement(True), draw_from_deck_with_replacement(True)


def draw_action(s, policy):
    # Draw an action given a state and policy distribution
    return HIT if rd.random() < policy[(s, HIT)] else STICK


def is_episode_terminated(r, a):
    return a == STICK or r == LOSE


def generate_episode(policy):
    s = generate_initial_state()
    a = draw_action(s, policy)

    while True:
        s1, r = step(s, a)
        a1 = draw_action(s1, policy)
        yield (s, a, r, s1, a1)

        # Episode ends after we stick or lose, whichever comes first
        if is_episode_terminated(r, a):
            break
        s = s1
        a = a1


def generate_initial_policy():
    # Generate the initial (random) ε-soft policy distribution
    policy = {}
    for i in range(1, NUMBER_OF_CARDS+1):
        for j in range(1, MAX_POINTS+1):
            s = i, j
            policy[(s, HIT)] = 0.5
            policy[(s, STICK)] = 0.5
    return policy


class Easy21MCControl(object):
    """
    Class encapsulating state for Monte-Carlo Control in Easy21
    """
    def __init__(self, T=100000, N0=100.0):
        """
        Initialiser.

        :param T: No. episodes (default 100000)
        :param N0: Initial value for N0 (default 100)
        """
        self.T = T
        self.N0 = N0
        self.policy = generate_initial_policy()
        self.N = defaultdict(int)
        self.Q = defaultdict(float)
        self.alpha = defaultdict(lambda: next(it.repeat(1.0)))
        self.eta = defaultdict(lambda: next(it.repeat(1.0)))
        self.V = []

    def build_V(self):
        for i in range(1, NUMBER_OF_CARDS+1):
            W = []
            for j in range(1, MAX_POINTS+1):
                s = i, j
                W.append(max(self.Q[(s, HIT)], self.Q[(s, STICK)]))
            self.V.append(W)

    def _update_state(self, episode, reward):
        for s, a, _, _, _ in episode:
            self.Q[(s, a)] += self.alpha[(s, a)] * (reward - self.Q[(s, a)])
            self.N[s] += 1
            self.N[(s, a)] += 1
        for s, a, _, _, _ in episode:
            self.alpha[(s, a)] = 1.0/self.N[(s, a)]
            self.eta[s] = self.N0/(self.N0 + self.N[s])

    def _adjust_policy(self, s):
        # Adjust policy in an ε-greedy fashion
        a_star = HIT if self.Q[(s, HIT)] >= self.Q[(s, STICK)] else STICK
        a_min = STICK if a_star == HIT else HIT
        eta = self.eta[s]
        self.policy[(s, a_star)] = 1 - eta + eta/len(ACTIONS)
        self.policy[(s, a_min)] = eta/len(ACTIONS)

    def run(self):
        """
        Run the evaluation. Attribute self.V contains V∗ (s) = max_a Q∗ (s, a)
        """
        t = 0
        while t < self.T:
            # Generate an episode and extract the (terminal) reward
            episode = [x for x in generate_episode(self.policy)]
            _, _, reward, _, _ = episode[-1]
            self._update_state(episode, reward)
            for s, _, _, _, _ in episode:
                self._adjust_policy(s)
            t += 1
        self.build_V()


if __name__ == '__main__':
    # Run Monte-Carlo Control in Easy21 with T=100,000,000 episodes and plot
    # the surface of V∗ (s) = max_a Q∗ (s, a)
    # WARNING: this could take some time to complete.
    easy21MCC = Easy21MCControl(T=int(1E8), N0=int(1E5))
    import datetime as dt
    s = dt.datetime.now()
    easy21MCC.run()

    t = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join('out', 'MC_V_{0:d}_{1:d}'.format(easy21MCC.T, easy21MCC.N0) + t + '.csv'), 'w') as f:
        f.writelines(map(lambda s: ','.join(map(str, s)) + '\n', easy21MCC.V))
    with open(os.path.join('out', 'MC_Q_{0:d}_{1:d}'.format(easy21MCC.T, easy21MCC.N0) + t + '.csv'), 'w') as f:
        f.writelines(map(lambda s: str(s) + '\n', easy21MCC.Q.items()))
    print('Elapsed time:', dt.datetime.now() - s)

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
import random as rd
import argparse

from constants import *
from question1 import step, draw_from_deck_with_replacement


def generate_initial_state():
    """
    Returns a random initial game state.

    :return: a tuple (a, b) where a is the dealer's initial card value, and b the player's
    """
    return draw_from_deck_with_replacement(True), draw_from_deck_with_replacement(True)


def draw_action(s, policy):
    """
    Draw an action given a state and policy distribution.

    :param s: a game state represented as a tuple (a, b) where a is the
              dealer's initial card value, and b the player's current card value
    :param policy: a policy pi(s, a)
    :return: HIT or STICK, whichever has the higher probability of
             occurrence in state s according to the policy.
    """
    return HIT if rd.random() < policy[(s, HIT)] else STICK


def is_episode_terminated(r, a):
    """
    Returns True if the an episode has terminated.

    :param r: the reward from the latest step in the episode
    :param a: the action taken
    :return: True if the an episode has terminated, False otherwise.
    """
    return a == STICK or r == LOSE


def generate_episode(policy):
    """
    Returns a generator which can be iterated to reveal
    successive game steps. Calling yield on this generator
    will return a 5-tuple: (s, a, r, s1, a1) where:
        s - the initial state
        a - the action taken in s
        r - the reward received
        s1 - the new state after a is performed
        a1 - a new action drawn from the policy in state s1

    :param policy: the policy pi(s, a)
    :return: a game episode generator
    """
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
    """
    Generate the initial (random) ε-soft policy distribution

    :return: a dict representing the policy pi(s, a) with
             pi(s, HIT) = pi(s, STICK) = 0.5 for all states s.
    """
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
    parser = argparse.ArgumentParser(description='Easy21MCControl cmd-line arguments.')

    parser.add_argument('--episodes', default=int(1E8), type=int)
    parser.add_argument('--N0', default=int(1E5), type=int)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--plot', default=None, type=str)

    args = parser.parse_args()

    def plot_v_surface(v):
        import pandas as pd
        df = pd.DataFrame(v)
        df.index = range(1, NUMBER_OF_CARDS+1)
        df.columns = range(1, MAX_POINTS+1)
        df = df.ix[:, range(12, 22)]

        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(1, 11), range(12, 22))
        ax.plot_surface (X, Y, df.values.T, cmap=cm.jet, rstride=1, cstride=1)
        plt.xlabel('Dealer Showing')
        plt.ylabel('Player Sum')
        plt.yticks(np.arange(12, 22, 2))
        plt.show()

    if args.plot is not None:
        import pandas as pd
        df = pd.DataFrame.from_csv(args.plot, index_col=None, header=None)
        plot_v_surface(df)
    else:
        T = args.episodes
        N0 = args.N0

        import datetime as dt
        s = dt.datetime.now()
        easy21MCC = Easy21MCControl(T=T, N0=N0)
        easy21MCC.run()

        if args.save:
            t = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            with open('mc_v_{0:d}_{1:d}_{2:s}.csv'.format(easy21MCC.T, easy21MCC.N0, t), 'w') as f:
                f.writelines(map(lambda s: ','.join(map(str, s)) + '\n', easy21MCC.V))
            with open('mc_q_{0:d}_{1:d}_{2:s}.csv'.format(easy21MCC.T, easy21MCC.N0, t), 'w') as f:
                f.writelines(map(lambda s: str(s) + '\n', easy21MCC.Q.items()))
        else:
            plot_v_surface(easy21MCC.V)

        print('Elapsed time:', dt.datetime.now() - s)

"""
RL Assignment - Easy21
@author: William Ferreira

We now consider a simple value function approximator using coarse coding. Use a binary
feature vector φ(s, a) with 3 * 6 * 2 = 36 features. Each binary feature has a value
of 1 iff (s, a) lies within the cuboid of state-space corresponding to that feature,
and the action corresponding to that feature. The cuboids have the following
overlapping intervals:

dealer(s) = {[1, 4], [4, 7], [7, 10]}
player(s) = {[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]}
a = {hit, stick}

where
• dealer(s) is the value of the dealer’s first card (1–10)
• sum(s) is the sum of the player’s cards (1–21)

Repeat the Sarsa(λ) experiment from the previous section, but using linear value function
approximation Q(s, a) = φ(s, a)⊤θ. Use a constant exploration of ε = 0.05 and a constant
step-size of 0.01. Plot the mean-squared error against λ. For λ = 0 and λ = 1 only,
plot the learning curve of mean-squared error against episode number.

                                                                            15 marks
"""
from collections import defaultdict
import itertools as it
import random as rd
import argparse

from constants import *
from question1 import step
from question2 import generate_initial_state, is_episode_terminated
from question3 import plot_mse


_dealer_features = [(1, 4), (4, 7), (7, 10)]
_player_features = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
_action_features = [HIT, STICK]
_feature_space = list(it.product(_dealer_features, _player_features, _action_features))


def _phi(s, a):
    """
    Implements φ(s, a) with 3 * 6 * 2 = 36 features.

    :param s: a game state represented as a tuple (a, b) where a is the
              dealer's initial card value, and b the player's current card value
    :param a: an action
    :return: list of indexes of those features which are switched on for s and a
    """
    dealer, player = s

    def contains(i, v):
        l, u = i
        return l <= v <= u

    def feature_on(f):
        d, p, act = f
        return contains(d, dealer) and contains(p, player) and act == a

    return [i for (i, f) in enumerate(_feature_space) if feature_on(f)]


class Easy21LinearFunctionApprox(object):
    """
    Class encapsulating state for Sarsa Control with
    function approximation in Easy21.
    """
    def __init__(self, lambda_, T=1000, eta=0.05, alpha=0.01):
        """
        Initializer

        :param lambda_: Value of trace weight parameter lambda
        :param T: No. of episodes (default 1000)
        :param eta: exploration probability (default 0.05)
        :param alpha: step-size (default 0.01)
        """
        self.lambda_ = lambda_
        self.T = T
        self.eta = eta
        self.alpha = alpha
        self.theta = [0]*len(_feature_space)
        self.e_trace = defaultdict(float)
        self.learning_curve = []

    def q(self, s, a):
        f_a = _phi(s, a)
        return sum([self.theta[i] for i in f_a])

    def _choose_action(self, s):
        if rd.random() < self.eta:
            return HIT if rd.random() < 0.5 else STICK
        f = [(a, sum([self.theta[i] for i in _phi(s, a)])) for a in ACTIONS]
        a, _ = max(f, key=lambda x: x[1])
        return a

    def _update_theta(self, delta):
        for i in range(len(_feature_space)):
            self.theta[i] += self.alpha * delta * self.e_trace[i]

    def extract_q(self):
        q = {}
        for i in range(1, NUMBER_OF_CARDS+1):
            for j in range(1, MAX_POINTS+1):
                s = i, j
                for a in ACTIONS:
                    q[(s, a)] = self.q(s, a)
        return q

    def run(self):
        """
        Run the evaluation.
        """
        t = 0

        while t < self.T:
            self.e_trace.clear()
            s = generate_initial_state()
            a = self._choose_action(s)

            while True:
                for i in range(len(_feature_space)):
                    self.e_trace[i] *= self.lambda_

                f_a = _phi(s, a)

                for i in f_a:
                    self.e_trace[i] += 1

                s1, r = step(s, a)
                delta = r - self.q(s, a)

                if is_episode_terminated(r, a):
                    self._update_theta(delta)
                    break

                s = s1
                a = self._choose_action(s)
                delta += self.q(s, a)
                self._update_theta(delta)
            self.learning_curve.append((t, self.extract_q()))
            t += 1
        return self


if __name__ == '__main__':
    # Run Linear Function Approximation for Easy21.
    parser = argparse.ArgumentParser(description='Easy21LinearFunctionApprox cmd-line arguments.')

    parser.add_argument('--episodes', default=int(1E3), type=int)
    parser.add_argument('--qdata', default='mc_q.csv', type=str)

    args = parser.parse_args()

    with open(args.qdata) as f:
        d = dict([eval(l) for l in f.readlines()])

    def compute_mse(q):
        mse = 0.0
        for x, v in d.items():
            v1 = q.get(x, 0.0)
            mse += (v - v1)**2
        return mse

    import datetime as dt
    s = dt.datetime.now()

    T = args.episodes

    lambdas = [x/10.0 for x in range(0, 11)]
    results = dict([(lm, Easy21LinearFunctionApprox(T=T, lambda_=lm).run()) for lm in lambdas])
    print('Elapsed time:', dt.datetime.now() - s)

    mse = [(lm, compute_mse(results[lm].extract_q())) for lm in lambdas]
    l0 = [(t+1, compute_mse(Q)) for (t, Q) in results[0].learning_curve]
    l1 = [(t+1, compute_mse(Q)) for (t, Q) in results[1].learning_curve]

    import numpy as np
    plot_mse(mse, l0, l1, np.arange(20, 50, 5), loc='upper right')


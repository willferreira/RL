"""
RL Assignment - Easy21
@author: William Ferreira

3 TD Learning in Easy21

Implement Sarsa(λ) in 21s. Initialise the value function to zero.
Use the same step-size and exploration schedules as in the previous
section. Run the algorithm with parameter values λ ∈ {0, 0.1, 0.2, ..., 1}.
Stop each run after 1000 episodes and report the mean-squared error over
all states s and actions a, comparing
the true values Q∗(s,a) computed in the previous section with the estimated
values Q(s, a) computed by Sarsa. Plot the mean- squared error against λ.

For λ = 0 and λ = 1 only, plot the learning curve of mean-squared error against
episode number.

                                                                15 marks
"""
from collections import defaultdict
import argparse
    
from constants import *
from question2 import Easy21MCControl, generate_episode, is_episode_terminated


class Easy21Sarsa(Easy21MCControl):
    """
    Class encapsulating state for Sarsa(lambda) Control in Easy21. Policy adjustment
    is inherited from Easy21MCControl.
    """
    def __init__(self, lambda_, T=1000, N0=100):
        """
        Initializer

        :param lambda_: Value of trace weight parameter lambda
        :param T: No. of episodes (default 1000)
        :param N0: Initial value for N0 (default 100)
        """
        super(Easy21Sarsa, self).__init__(T=T, N0=N0)

        self.lambda_ = lambda_
        self.e_trace = defaultdict(float)
        self.learning_curve = []

    def _update_state(self, s, a, r, s1, a1):
        q = self.Q[(s1, a1)] if not is_episode_terminated(r, a) else 0.0
        delta = r + q - self.Q[(s, a)]
        self.e_trace[(s, a)] += 1
        for i in range(1, NUMBER_OF_CARDS+1):
            for j in range(1, MAX_POINTS+1):
                st = i, j
                for act in ACTIONS:
                    self.Q[(st, act)] += self.alpha[(st, act)] * delta * self.e_trace[(st, act)]
                    self.e_trace[(st, act)] *= self.lambda_
        self.N[s] += 1
        self.N[(s, a)] += 1
        self.alpha[(s, a)] = 1/self.N[(s, a)]
        self.eta[s] = self.N0/(self.N0 + self.N[s])

    def run(self):
        """
        Run the evaluation.
        """
        t = 0
        while t < self.T:
            # Reset eligibility traces
            self.e_trace.clear()

            # Generate an episode and process each step on-the-fly
            for s, a, r, s1, a1 in generate_episode(self.policy):
                self._update_state(s, a, r, s1, a1)
                self._adjust_policy(s)
            self.learning_curve.append((t, dict(self.Q)))
            t += 1
        return self


def plot_mse(mse, lambda0, lambda1, scale, loc='lower right'):
    """
    Draw a plot of the MSE against lambda. Draw a plot of the
    MSE of the learning curve for lambda = 0,1.

    :param mse: MSE versus lambda
    :param lambda0: lambda=0 learning curve MSE
    :param lambda1: lambda=1 learning curve MSE
    :param scale: scale for y-axis
    :param loc: location of legend
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(*zip(*mse))
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    plt.yticks(scale)

    ax = fig.add_subplot(212)
    ax.plot(*zip(*lambda0), label='$\lambda=0$')
    plt.xlabel('Episode')
    plt.ylabel('MSE')
    ax.plot(*zip(*lambda1), label='$\lambda=1$')
    plt.legend(loc=loc)

    plt.show()


if __name__ == '__main__':
    # Run Sarsa(lambda) for Easy21.
    parser = argparse.ArgumentParser(description='Easy21Sarsa cmd-line arguments.')

    parser.add_argument('--episodes', default=int(1E3), type=int)
    parser.add_argument('--N0', default=int(1E2), type=int)
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
    N0 = args.N0

    lambdas = [x/10.0 for x in range(0, 11)]
    results = dict([(lm, Easy21Sarsa(lambda_=lm, T=T, N0=N0).run()) for lm in lambdas])
    print('Elapsed time:', dt.datetime.now() - s)

    mse = [(lm, compute_mse(results[lm].Q)) for lm in lambdas]
    l0 = [(t+1, compute_mse(Q)) for (t, Q) in results[0].learning_curve]
    l1 = [(t+1, compute_mse(Q)) for (t, Q) in results[1].learning_curve]

    import numpy as np
    plot_mse(mse, l0, l1, np.arange(0, 250, 25))
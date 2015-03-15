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
import sys
import os
from collections import defaultdict
    
from constants import *
from question2 import Easy21MCControl, generate_episode, is_episode_terminated


class Easy21Sarsa(Easy21MCControl):
    """
    Class encapsulating state for Sarsa(lambda) Control in Easy21
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


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        d = dict([eval(l) for l in f.readlines()])

    def compute_mse(q):
        mse = 0.0
        for x, v in d.items():
            v1 = q.get(x, 0.0)
            mse += (v - v1)**2
        return mse

    import datetime as dt
    s = dt.datetime.now()

    T = 10000
    N0 = 100
    lambdas = [x/10.0 for x in range(0, 11)]
    results = dict([(lm, Easy21Sarsa(lambda_=lm, T=T, N0=N0).run()) for lm in lambdas])
    mse = [(lm, compute_mse(results[lm].Q)) for lm in lambdas]

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    def write_mse(f, mse):
        f.writelines(['{0:.1f},{1:.2f}\n'.format(x, y) for x, y in mse])

    with open(os.path.join('out', 'SARSA_MSE_{0:d}_{1:d}'.format(T, N0) + ts + '.csv'), 'w') as f:
        write_mse(f, mse)

    def write_mse_lambda(lm):
        with open(os.path.join('out', 'SARSA_MSE_lambda{0:d}_{1:d}_{2:d}'.format(int(lm), T, N0) + ts + '.csv'), 'w') as f:
            write_mse(f, [(t+1, compute_mse(Q)) for (t, Q) in results[lm].learning_curve])

    write_mse_lambda(0)
    write_mse_lambda(1)

    print('Elapsed time:', dt.datetime.now() - s)
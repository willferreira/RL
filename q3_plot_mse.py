import sys
import os

import numpy as np

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def get_mse(name):
        with open(os.path.join('out', name + sys.argv[1]), 'r') as f:
            mse = [eval(l) for l in f.readlines()]
        return mse

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(*zip(*get_mse('SARSA_MSE_')))
    plt.xlabel('lambda')
    plt.ylabel('MSE')
    plt.yticks(np.arange(0, 35, 5.0))

    ax = fig.add_subplot(212)
    ax.plot(*zip(*get_mse('SARSA_MSE_lambda0_')), label='lambda=0')
    plt.xlabel('Episode')
    plt.ylabel('MSE')
    plt.legend()

    ax.plot(*zip(*get_mse('SARSA_MSE_lambda1_')), label='lambda=1')
    plt.legend()

    plt.show()

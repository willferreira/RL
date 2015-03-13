import sys
import os

import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.DataFrame.from_csv(os.path.join('out', sys.argv[1]), index_col=None, header=None)
    df.index = range(1, 11)
    df.columns = range(1, 22)

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(1,11), range(1, 22))
    ax.plot_surface (X, Y, df.values.T, cmap=cm.jet, rstride=1, cstride=1)
    plt.xlabel('Dealer Showing')
    plt.ylabel('Player Sum')
    plt.show()

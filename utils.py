import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

def plot_scores(scores):
    # plot the scores
    fig = plt.figure(figsize=(20, 10))

    x = np.arange(len(scores))
    y = scores

    plt.plot(x, y)

    x_sm = np.array(x)
    y_sm = np.array(y)

    x_smooth = np.linspace(x_sm.min(), x_sm.max(), 20)
    y_smooth = spline(x, y, x_smooth)
    plt.plot(x_smooth, y_smooth, 'orange', linewidth=4)

    plt.ylabel('Score')
    plt.xlabel('Episode #')

    plt.ylim(ymin=0)

    plt.show()

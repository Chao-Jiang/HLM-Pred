import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def print_stats(distances, title=None, draw=True, show=True):
    print(distances)
    from scipy import stats
    n, min_max, mean, var, skew, kurt = stats.describe(np.asarray(distances))
    median = np.median(distances)
    first_quartile = np.percentile(distances, 25)
    third_quartile = np.percentile(distances, 75)
    print('\nDistances statistics:')
    print("Minimum: {0:9.4f} Maximum: {1:9.4f}".format(min_max[0], min_max[1]))
    print("Mean: {0:9.4f}".format(mean))
    print("Variance: {0:9.4f}".format(var))
    print("Median: {0:9.4f}".format(median))
    print("First quartile: {0:9.4f}".format(first_quartile))
    print("Third quartile: {0:9.4f}".format(third_quartile))
    if draw:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.set_style("whitegrid")
            plt.figure()
            vio_ax = sns.violinplot(x=distances, cut=0)
            vio_ax.set_xlabel('distances_error')
            if title is not None:
                plt.title(title)
            plt.figure()
            strip_ax = sns.stripplot(x=distances)
            strip_ax.set_xlabel('distances_error')
            if title is not None:
                plt.title(title)
            if show:
                plt.show()
        except ImportError:
            pass

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

def plot_3d(points_list, title=None, draw_now=True, seq_length=None, start=0):
    assert len(points_list) <= 2, 'Length of points list should not be greater than two'
    fig = plt.figure()
    plt.rcParams['axes.facecolor'] = 'white'
    ax = fig.add_subplot(111, projection='3d')
    for idx in xrange(points_list[0].shape[0]):
        if seq_length is None:
            ax.plot(points_list[0][idx, :, 0],
                    points_list[0][idx, :, 1],
                    points_list[0][idx, :, 2],
                    'b-', label='input')
        else:
            import numbers
            assert isinstance(seq_length, numbers.Integral), 'Sequence length must be integer if provided'
            # if start > 0:
            #     ax.plot(points_list[0][idx, 0: start+1, 0],
            #             points_list[0][idx, 0: start+1, 1],
            #             points_list[0][idx, 0: start+1, 2],
            #             'm-', linewidth=1, label='true_trajectory')
            ax.plot(points_list[0][idx, :, 0],
                    points_list[0][idx, :, 1],
                    points_list[0][idx, :, 2],
                    'm-', linewidth=1, label='true_trajectory')
            ax.plot(points_list[0][idx, start: start+seq_length, 0],
                    points_list[0][idx, start: start+seq_length, 1],
                    points_list[0][idx, start: start+seq_length, 2],
                    'b-', linewidth=1, label='input')
            # ax.plot(points_list[0][idx, start+seq_length-1:, 0],
            #         points_list[0][idx, start+seq_length-1:, 1],
            #         points_list[0][idx, start+seq_length-1:, 2],
            #         'm-', linewidth=1, label='true_trajectory')
        ax.scatter(points_list[0][idx, :, 0],
                   points_list[0][idx, :, 1],
                   points_list[0][idx, :, 2],
                   marker='o', c='r', s=15)
        # Get rid of the panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if len(points_list) == 2:
            if len(points_list[1].shape) == 2:
                ax.scatter(points_list[1][idx, 0],
                           points_list[1][idx, 1],
                           points_list[1][idx, 2],
                           c='k', marker='x',
                           label='prediction')
            elif len(points_list[1].shape) == 3:
                ax.scatter(points_list[1][idx, :, 0],
                           points_list[1][idx, :, 1],
                           points_list[1][idx, :, 2],
                           c='k', marker='x',
                           label='prediction')
                ax.plot(points_list[1][idx, :, 0],
                        points_list[1][idx, :, 1],
                        points_list[1][idx, :, 2],
                        'kx-', linewidth=1)
    ax.legend(ncol=1, prop={'size': 12})
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    set_aspect_equal_3d(ax)
    if title is not None:
        plt.title(title)
    if draw_now:
        plt.show()


'''
@motjuste
Basic plotting using matplotlib
'''
import matplotlib.pyplot as plt


def plot2d(X, axs=None, show=True, set_aspect_equal=True, x_lim=None, y_lim=None, show_axes_through_origin=False, plotlabel=None, title=None):

    # FIXME: @motjuste: proper asserts, and testing, please

    if axs is None:
        # create a new figure and axes
        fig = plt.figure()
        axs = fig.add_subplot(111)

    # meat and potato of plotting
    axs.plot(X[0,:], X[1, :], 'o', label=plotlabel)

    # Garnishing starts here
    if title is not None:
        axs.set_title(title)

    leg = axs.legend()
    leg.get_frame().set_alpha(0.5)

    if set_aspect_equal:
        axs.set_aspect('equal')

    if x_lim is not None:
        assert len(x_lim) == 2, "mismatch in x_lim"
        axs.set_xlim(x_lim[0], x_lim[1])

    if y_lim is not None:
        assert len(y_lim) == 2, "mismatch in y_lim"
        axs.set_ylim(y_lim[0], y_lim[1])

    if show_axes_through_origin:
        axs.axhline(y=0)
        axs.axvline(x=0)

    # IPythonNotebook will handle showing if you want
    if show:
        plt.show()

'''
@motjuste : 22-Nov-2015

- Code relevant to Task 1.4
'''
import numpy as np
import matplotlib.pyplot as plt


def plot_Lp_unit_circle(p=2, atol=0.001):
    x = np.linspace(0, 1, 2000)

    y = np.power((1 - np.power(x, p)), 1/p)

    # plot the results
    plt.figure()
    plt.plot(x, y)

    # since the unit circle is symmetric in all quadrants
    # the plots in different quadrants would appear in different colors
    # this is intentional
    plt.plot(-x, y)
    plt.plot(x, -y)
    plt.plot(-x, -y)

    # set title of plot to show p
    plt.title("p=%.1f" % (p))

    # make the plot aspect ratio square and set range
    plt.axis('equal')
    plt.axis([-1.2, 1.2, -1.2, 1.2])

    # add axes through the origin and grids
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.grid()

    plt.show()


def plot_Lp_unit_circle_linalg(p=2, atol=0.001):
    x = np.linspace(-1, 1, 2000)
    y = np.linspace(-1, 1, 2000)

    # for convenience
    m = np.meshgrid(x, y)

    norms = np.linalg.norm(m, axis=0, ord=p)
    # FIXME: the following allows norms > 1 to the atol
    xi, yi = np.where(np.isclose(norms, 1, atol=atol))

    plt.plot(x[xi], y[yi], ".", )
    plt.title("p=%.1f" % (p))

    # adding the axes through origin and grids
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.grid()

    # the plots may not have a square's aspect ratio
    plt.axis('equal')
    plt.axis([-1.2, 1.2, -1.2, 1.2])

    plt.show()

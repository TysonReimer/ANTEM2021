"""
Tyson Reimer
University of Manitoba
July 09th, 2020
"""

import matplotlib.pyplot as plt

###############################################################################


def init_plt(figsize=(12, 6), labelsize=18):
    """

    Parameters
    ----------
    figsize : tuple
        The figure size
    labelsize : int
        The labelsize for the axis-ticks
    """

    plt.figure(figsize=figsize)
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=labelsize)

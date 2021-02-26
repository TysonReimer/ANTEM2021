"""
Tyson Reimer
University of Manitoba
June 19th, 2019
"""

import pickle
import numpy as np

###############################################################################


def load_pickle(path):
    """Loads a pickle file

    Parameters
    ----------
    path : str
        The full path to the .pickle file that will be loaded

    Returns
    -------
    loaded_var :
        The loaded variable, can be array_like, int, str, dict_to_save,
        etc.
    """

    with open(path, 'rb') as handle:
        loaded_var = pickle.load(handle)

    return loaded_var


def save_pickle(var, path):
    """Saves the var to a .pickle file at the path specified

    Parameters
    ----------
    var : object
        A variable that will be saved
    path : str
        The full path of the saved .pickle file
    """

    with open(path, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_birrs_txt(txt_path):
    """Loads a raw .txt file generated by the BIRRS system from a scan

    Loads the file with path txt_path and returns the data stored in that
    arr in the time or frequency domain. Assumes the usual format produced
    by the BIRRS software as of May 7, 2019: for N antenna positions,
    the raw data file contains 2N columns. The first column contains the
    real part of the scattering parameter for the first antenna position,
    the second column contains the imag part, the third column contains
    the real part for the second antenna position, etc.

    Parameters
    ----------
    txt_path : str
        The complete path to the file containing the raw data from a
        measured scan, in the BIRRS software format as of May 7, 2019

    Returns
    -------
    data : array_like
        2D arr containing the measured radar signal for each antenna position
        used in the scan (in freq-domain)
    """

    # Load the data
    raw_data = np.genfromtxt(txt_path, dtype='float', delimiter='')

    # Find the number of frequencies and scan positions
    n_freqs, n_ant_pos = raw_data.shape
    n_ant_pos //= 2

    # Initialize arr for storing the frequency domain data
    fd_data = np.zeros([n_freqs, n_ant_pos], dtype=complex)

    for ant_pos in range(n_ant_pos):

        # Combine the real and imaginary parts of the scattering
        # parameter, as they're stored in the file
        fd_data[:, ant_pos] = (raw_data[:, 2 * ant_pos]
                               + 1j * raw_data[:, 2 * ant_pos + 1])

    else:  # Leave data in the frequency domain
        data = fd_data

    return data

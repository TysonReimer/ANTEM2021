"""
Tyson Reimer
University of Manitoba
February 12th, 2021
"""

import os
import numpy as np

from umbms import get_proj_path, verify_path

from umbms.loadsave import save_pickle, load_birrs_txt

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/raw/')

__OUT_DIR = os.path.join(get_proj_path(), 'data/fd/')
verify_path(__OUT_DIR)

###############################################################################

if __name__ == "__main__":

    n_expts = 8  # Hard-code because we know 8 scans performed
    n_freqs = 1001  # Hard-code because 1001 freqs used
    n_ant_pos = 72  # Hard-code because 72 antenna positions used

    # Init arrays to store s11 and s21 data
    fd_s11 = np.zeros([n_expts, n_freqs, n_ant_pos], dtype=complex)
    fd_s21 = np.zeros([n_expts, n_freqs, n_ant_pos], dtype=complex)

    for ii in range(n_expts):  # For each experiment

        # Load this s11 .txt file
        fd_s11[ii, :, :] = \
            load_birrs_txt(os.path.join(__DATA_DIR,
                                        's11_expt%d.txt' % (ii + 1)))

        # Load this s21 .txt file
        fd_s21[ii, :, :] = \
            load_birrs_txt(os.path.join(__DATA_DIR,
                                        's21_expt%d.txt' % (ii + 1)))

    # Save to .pickle files
    save_pickle(fd_s11, os.path.join(__OUT_DIR, 's11_fd.pickle'))
    save_pickle(fd_s21, os.path.join(__OUT_DIR, 's21_fd.pickle'))

"""
Tyson Reimer
University of Manitoba
February 12th, 2021
"""

import os
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.beamform.sigproc import iczt

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/fd/')

__OUT_DIR = os.path.join(get_proj_path(), 'output/figs/')
verify_path(__OUT_DIR)

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    n_expts = 8  # Hard-code because we know 8 scans performed
    n_freqs = 1001  # Hard-code because 1001 freqs used
    n_ant_pos = 72  # Hard-code because 72 antenna positions used

    # Define parameters for time domain conversion
    n_ts = 700  # Number of time-points to use
    ini_t = 0.5e-9  # Initial time point to use
    fin_t = 5.5e-9  # Final time point
    ini_f = 1e9  # Initial scan frequency
    fin_f = 8e9  # Final scan frequency

    # Load S11 frequency domain data
    fd_s11 = load_pickle(os.path.join(__DATA_DIR, 's11_fd.pickle'))

    # Init array for storing time domain S11
    td_data = np.zeros([np.size(fd_s11, axis=0),
                        n_ts,
                        np.size(fd_s11, axis=2)],
                       dtype=complex)

    # Convert to the time domain
    for ii in range(np.size(fd_s11, axis=0)):

        # Use ICZT to convert to the time-domain
        td_data[ii, :, :] = iczt(fd_s11[ii, :, :], ini_t=ini_t, fin_t=fin_t,
                                 ini_f=ini_f, fin_f=fin_f, n_time_pts=n_ts)

    # Indices of expts containing saline-filled tumour shell
    sal_tum_idxs = [2, 5]

    # Indices of expts containing glycerin-filled tumour shell
    gly_tum_idxs = [1, 4]

    # Indices of expts of empty adipose shell
    adi_idxs = [0, 3, 6]

    # Index of empty-reference scan
    emp_idx = [7]

    # Get colormap for plots
    viridis = get_cmap('viridis')

    # Init vars for inset plot to avoid Warnings
    inset_gly = 0
    inset_sal = 0
    inset_adi1 = 0
    inset_adi2 = 0

    for adi_id in adi_idxs:  # For each reference scan

        print('\tAdi ID %d reference...' % adi_id)

        ref_td = td_data[adi_id, :, :]  # Grab the reference scan

        for sal_id in sal_tum_idxs:  # For each saline tumour

            logger.info('\t\tSal ID %d target...' % sal_id)

            # Get and calibrate the saline tumour scan data
            sal_cal = np.abs(td_data[sal_id, :, :] - ref_td)

            # Get and calibrate the glycerin tumour scan data
            gly_td = np.abs(td_data[sal_id - 1, :, :] - ref_td)

            # Define thresholds for defining the ROI
            thresholds = np.linspace(0, 1, 100)

            # Init arr for storing the number of pixels in each ROI
            n_pixels = np.zeros_like(thresholds)

            # Init arrays for storing means and standard deviations
            # in all ROIs
            sal_means = np.zeros_like(thresholds)
            sal_stds = np.zeros_like(thresholds)
            gly_means = np.zeros_like(thresholds)
            gly_stds = np.zeros_like(thresholds)
            adi1_means = np.zeros_like(thresholds)
            adi1_stds = np.zeros_like(thresholds)
            adi2_means = np.zeros_like(thresholds)
            adi2_stds = np.zeros_like(thresholds)
            all_adi_means = np.zeros_like(thresholds)
            all_adi_stds = np.zeros_like(thresholds)

            # For each threshold used to define the ROI
            for ii in range(len(thresholds)):

                # Get the ROI
                roi = sal_cal >= thresholds[ii] * np.max(sal_cal)

                # Get the pixels from each sinogram
                sal_pix = sal_cal[roi]
                gly_pix = gly_td[roi]

                n_pixels[ii] = np.size(sal_pix)  # Store n_pixels

                # Store averages and standard deviations
                sal_means[ii], sal_stds[ii] = np.mean(sal_pix), np.std(sal_pix)
                gly_means[ii], gly_stds[ii] = np.mean(gly_pix), np.std(gly_pix)

                adi_pixs = []  # Init list for storing adipose pixels

                cc = 0  # Init counter
                for other_adi in adi_idxs:  # For each adipose scan

                    # If this adi scan was not the reference here
                    if other_adi != adi_id:

                        # Calibrate this scan
                        adi_td = np.abs(td_data[other_adi, :, :] - ref_td)

                        # Store the pixels in the ROI
                        adi_pix = adi_td[roi]
                        adi_pixs.append(adi_pix)

                        # Store results to arrays
                        if cc == 0:
                            adi1_means[ii], adi1_stds[ii] = (np.mean(adi_pix),
                                                             np.std(adi_pix))
                        else:
                            adi2_means[ii], adi2_stds[ii] = (np.mean(adi_pix),
                                                             np.std(adi_pix))

                        cc += 1

                # If this threshold is halfway between 0 and 1
                if ii == len(thresholds) // 2:

                    # Store the ROI distributions for the inset plot
                    inset_gly = gly_pix
                    inset_sal = sal_pix
                    inset_adi1 = adi_pixs[0]
                    inset_adi2 = adi_pixs[1]

                # Store the distribution of adipose pixels with the
                # higher average to compare to glycerin-tumour
                if adi1_means[ii] > adi2_means[ii]:
                    all_adi_means[ii] = adi1_means[ii]
                    all_adi_stds[ii] = adi1_stds[ii]
                else:
                    all_adi_means[ii] = adi2_means[ii]
                    all_adi_stds[ii] = adi2_stds[ii]

            # Do statistical t-test
            ts, ps = stats.ttest_ind_from_stats(mean1=gly_means, std1=gly_stds,
                                                nobs1=n_pixels,
                                                mean2=all_adi_means,
                                                std2=all_adi_stds,
                                                nobs2=n_pixels,
                                                equal_var=False,
                                                alternative='less')

            # Make figure for the t-test
            plt.figure(figsize=(12, 6))
            plt.rc('font', family='Times New Roman')
            plt.tick_params(labelsize=20)
            plt.plot(thresholds, ps, 'ko--')
            plt.title("p-Values for ROI Defined by"
                      " Saline Tumour\nResponse, Scan %d,"
                      " with Scan %d Reference"
                      % (sal_id, adi_id), fontsize=24)
            plt.xlabel('Threshold of Maximum Used to Define ROI', fontsize=22)
            plt.ylabel('p-Value from 1-sided t-Test', fontsize=22)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(os.path.join(__OUT_DIR,
                                     'pValues_sal_%d_ref_%d.png'
                                     % (sal_id, adi_id)),
                        dpi=300, transparent=False)

            # Plot the average S11 in the ROI for each threshold
            fig = plt.figure(figsize=(12, 6))
            plt.rc('font', family='Times New Roman')
            plt.tick_params(labelsize=18)
            plt.plot(thresholds, sal_means, label='Saline Tumour',
                     color=viridis(0))
            plt.fill_between(thresholds, y1=sal_means - sal_stds,
                             y2=sal_means + sal_stds, color=viridis(0),
                             alpha=0.3)

            plt.plot(thresholds, gly_means, label='Glycerin Tumour',
                     color=viridis(0.3))
            plt.fill_between(thresholds, y1=gly_means - gly_stds,
                             y2=gly_means + gly_stds, color=viridis(0.3),
                             alpha=0.3)
            plt.plot(thresholds, adi1_means, label='Homogeneous Phantom 1',
                     color=viridis(0.6))
            plt.fill_between(thresholds, y1=adi1_means - adi1_stds,
                             y2=adi1_means + adi1_stds, color=viridis(0.6),
                             alpha=0.3)

            plt.plot(thresholds, adi2_means, label='Homogeneous Phantom 2',
                     color=viridis(0.9))
            plt.fill_between(thresholds, y1=adi2_means - adi2_stds,
                             y2=adi2_means + adi2_stds, color=viridis(0.9),
                             alpha=0.3)

            plt.xlabel('Threshold of Maximum Value used to Define ROI',
                       fontsize=20)
            plt.ylabel(r'Average S$_{\mathdefault{11}}$ Response', fontsize=20)
            plt.xlim([0, 1])
            plt.legend(fontsize=20)
            plt.ylim([0, 1.1 * np.max(sal_means)])
            plt.axvline(x=0.5, linestyle='--', color='k', lw=1)
            plt.tight_layout()

            # Make the inset distribution plot
            ax_in = fig.add_axes([0.65, 0.325, 0.3, 0.3])
            sns.distplot()
            sns.distplot(inset_sal, label='Saline Tumour',
                         color=viridis(0),
                         hist_kws={'alpha': 0.4})
            sns.distplot(inset_gly, label='Glycerin Tumour',
                         color=viridis(0.3),
                         hist_kws={'alpha': 0.4})
            sns.distplot(inset_adi1, label='Homogeneous Phantom 1',
                         color=viridis(0.6), hist_kws={'alpha': 0.4})
            sns.distplot(inset_adi2, label='Homogeneous Phantom 2',
                         color=viridis(0.9), hist_kws={'alpha': 0.4})
            ax_in.set_xlabel(r'Time-Domain |S$_{\mathdefault{11}}$|',
                             fontsize=14)
            ax_in.set_ylabel('Kernel Density Estimate', fontsize=14)
            ax_in.set_xlim([0, 1.1 * np.max(sal_means)])
            plt.show()
            plt.savefig(os.path.join(__OUT_DIR,
                                     'threshold_plt_sal_%d_ref_%d.png'
                                     % (sal_id, adi_id)),
                        dpi=300, transparent=False)

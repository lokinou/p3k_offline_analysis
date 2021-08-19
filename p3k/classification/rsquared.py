
import mne
import numpy as np
import seaborn as sns
import pandas as pd

from typing import Tuple
import matplotlib.pyplot as plt

def signed_r_square(epochs: mne.Epochs,
                    time_epoch: Tuple[float],
                    display_rsq_plot: bool = True):
    rsq = signed_r_square_mne(epochs, classes=['Target', 'NonTarget'])
    # make a pandas database to properly display electrodes and samples
    fs = epochs.info['sfreq']
    x = np.float64(list(range(rsq.shape[1]))) * (1000 / fs)
    x = x.round(decimals=0).astype(np.int64) + np.int64(time_epoch[0] * 1000)
    df_rsq = pd.DataFrame(rsq, columns=x, index=epochs.info['ch_names'])

    if display_rsq_plot:
        fig_rsq = plt.figure()
        ax = sns.heatmap(df_rsq, linewidths=0, cmap="coolwarm").set(title='Signed r-square maps Target vs Non-Target',
                                                              xlabel='Time (ms)')
    else:
        fig_rsq = None

    return rsq, fig_rsq


# From https://github.com/bbci/wyrm/blob/master/wyrm/processing.py
# Bastian Venthur for wyrm
# Code initially from Benjamin Blankertz for bbci (Matlab)

def signed_r_square_mne(epochs, classes=[0, 1], classaxis=0, **kwargs):
    """Calculate the signed r**2 values.
    This method calculates the signed r**2 values over the epochs of the
    ``dat``.
    Parameters
    ----------
    epochs : MNE epoched data
    classes: list, optional
        (either int index or str for the class name of the epoch))
    classaxis : int, optional
        the dimension containing epochs
    Returns
    -------
    signed_r_square : ndarray
        the signed r**2 values, signed_r_square has one axis less than
        the ``dat`` parameter, the ``classaxis`` has been removed
    Examples
    --------
    >>> dat.data.shape
    (400, 100, 64)
    >>> r = calculate_signed_r_square(dat)
    >>> r.shape
    (100, 64)
    """
    # TODO: explain the algorithm in the docstring and add a reference
    # to a paper.
    # select class 0 and 1
    # TODO: make class 0, 1 variables
    fv1 = epochs[classes[0]]._data
    fv2 = epochs[classes[1]]._data
    # number of epochs per class
    l1 = epochs[classes[0]]._data.shape[classaxis]
    l2 = epochs[classes[1]]._data.shape[classaxis]
    # calculate r-value (Benjamin approved!)
    a = (fv1.mean(axis=classaxis) - fv2.mean(axis=classaxis)) * np.sqrt(l1 * l2)
    b = epochs._data.std(axis=classaxis) * (l1 + l2)
    r = a / b
    # return signed r**2
    return np.sign(r) * np.square(r)




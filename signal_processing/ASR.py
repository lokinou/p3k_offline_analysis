import mne
import numpy as np
import matplotlib.pyplot as plt
from meegkit.asr import ASR

def train_asr(raw: mne.io.BaseRaw) -> ASR:
    fs = int(raw.info["sfreq"])  # sampling frequency
    method = 'riemann'  # if error, use 'euclid' (still not supported out of the box 04.07.2021!)
    method = 'euclid'  # pymanopt library still buggy
    window_s = .5  # .5 sec window of analysis
    data_interval_s = None  # (begin, end) in sec of the training sample
    #estimator = 'lwf'  #leave blank if using euclidian mode

    # define the ASR model using riemannian method
    #asr_model = ASR(sfreq=fs, method=method, win_len=window_s, estimator=estimator)

    # if failing (after trying twice. SVD error occurs for no reason sometimes)
    asr_model = ASR(sfreq=fs, method=method, win_len=window_s)

    # The best would be to choose another recording during the same session to train the model without overfitting
    data = raw._data.copy()  # the numpy array with data is stored in the _data variable

    # Select a time interval for training data
    train_idx = None
    if data_interval_s is not None:
        train_idx = np.arange(data_interval_s[0] * fs, data_interval_s[1] * fs, dtype=int)
    # otherwise select the whole training set
    else:
        train_idx = np.arange(0, data.shape[1])

    train_data = data[:, train_idx]
    print('Training on samples of size {}'.format(train_data.shape))

    # fir the ASR model with data intervals
    _, sample_mask = asr_model.fit(train_data)
    print('Model trained')

    return asr_model

def apply_asr(raw: mne.io.RawArray, asr_model: ASR, display_plot: bool = False):
    clean = asr_model.transform(raw._data.copy())

    display_window_s = 10  #
    fs = np.int(raw.info['sfreq'])
    if display_plot:  #
        data_p = raw._data[:, 0:fs * display_window_s]  # reshape to (n_chans, n_times)
        clean_p = clean[:, 0:fs * display_window_s]

        ###############################################################################
        # Plot the results
        # -----------------------------------------------------------------------------
        #
        # Data was trained on a 40s window from 5s to 45s onwards (gray filled area).
        # The algorithm then removes portions of this data with high amplitude
        # artifacts before running the calibration (hatched area = good).
        nb_ch_disp = len(raw.info['ch_names'])
        times = np.arange(data_p.shape[-1]) / fs
        f, ax = plt.subplots(nb_ch_disp, sharex=True, figsize=(16, 8))
        for i in range(nb_ch_disp):
            # ax[i].fill_between(train_idx / fs, 0, 1, color='grey', alpha=.3,
            #                   transform=ax[i].get_xaxis_transform(),
            #                   label='calibration window')
            # ax[i].fill_between(train_idx / fs, 0, 1, where=sample_mask.flat,
            #                   transform=ax[i].get_xaxis_transform(),
            #                   facecolor='none', hatch='...', edgecolor='k',
            #                   label='selected window')
            ax[i].plot(times, data_p[i], lw=.5, label='before ASR', color='r')
            ax[i].plot(times, clean_p[i], label='after ASR', lw=.5, color='gray')
            # ax[i].plot(times, raw[i]-clean[i], label='Diff', lw=.5)
            # ax[i].set_ylim([-50, 50])
            ax[i].set_ylabel(f'ch{i}')
            ax[i].set_yticks([])
        ax[i].set_xlabel('Time (s)')
        ax[0].legend(fontsize='small', bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.subplots_adjust(hspace=0, right=0.75)
        plt.suptitle('Before/after ASR')
        plt.show()
    raw.data_ = clean

    return raw

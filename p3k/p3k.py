from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Union
import os
import re
import mne.io
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis

from meegkit.asr import ASR

from p3k import openvibe
from p3k import bci2000
from p3k.wyrm import signed_r_square_mne

from sklearn import model_selection
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

@dataclass
class SpellerInfo:
    nb_stimulus_rows: int = None
    nb_stimulus_cols: int = None
    nb_seq: int = None

@dataclass
class DisplayPlots:
    raw: bool = False
    infinite_reference: bool = False
    bandpassed: bool = False
    asr: bool = True
    csd: bool = False
    cross_correlation: bool = False
    epochs: bool = True
    reject_epochs: bool = True
    butterfly: bool = False
    butterfly_topomap: bool = True
    channel_average: bool = True
    erp_heatmap: bool = False
    erp_heatmap_channelwise: bool = False
    signed_r_square: bool = True

TARGET_MAP = {'0':0, '1':1, '10':10}


def make_output_folder(filename_s: Union[str, List[str]], fig_folder: str) -> str:
    if isinstance(filename_s, list):
        output_name = Path(filename_s[0]).stem
    else:
        output_name = Path(filename_s).stem
        if len(filename_s) > 1:
            output_name = output_name + f'_{len(filename_s)}_files'
    print('Figures will have the name: {}'.format(output_name))

    fig_folder = os.path.join(fig_folder, output_name)
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)
    print('Created output directory'.format(fig_folder))

    return output_name


def define_channels(raw: mne.io.BaseRaw,
                    channel_names: List[str],
                    montage: mne.channels.DigMontage = None):
    print("Channel names from data: {}".format(raw.info['ch_names']))

    if montage is None:
        montage = mne.channels.make_standard_montage('standard_1005')

    if channel_names is None:
        channel_names = raw.info['ch_names']
        print('Using channel names directly from the data files: {}'.format(channel_names))
    elif len(channel_names) < len(raw.info['ch_names']):

        # append missing channels to cname
        channel_names.extend(raw.info['ch_names'][len(channel_names) - len(raw.info['ch_names']):])
        print('Using user defined channel names and ignoring other channels: {}'.format(channel_names))
    elif len(channel_names) > len(raw.info['ch_names']):
        print('Number of channels in data (n={}) is lower than the declared channel names {}'.format(
            len(raw.info['ch_names']), channel_names))
        raise
    else:
        print('Using user defined channel names: {}'.format(channel_names))

    nb_chan = len(raw.info['ch_names'])
    nb_def_ch = len(channel_names)

    # regular expressions to look for certain channel types
    eog_pattern = re.compile('eog|EOG')  # EOG electrodes
    emg_pattern = re.compile('emg|EMG')  # EMG electrodes
    mastoid_pattern = re.compile('^[aA][0-9]+')  # A1 and A2 electrodes (mastoids)

    types = []
    cname_map = dict(zip(raw.info['ch_names'], channel_names))
    for nc in channel_names:
        if eog_pattern.search(nc) is not None:
            t = 'eog'
        elif emg_pattern.search(nc) is not None:
            t = 'emg'
        elif mastoid_pattern.search(nc) is not None:
            t = 'misc'
        elif nc in montage.ch_names:  # check the 10-05 montage for all electrode names
            t = 'eeg'
        else:
            t = 'misc'  # if not found, discard the channel as misc

        types.append(t)

    type_map = dict(zip(channel_names, types))

    # rename and pick eeg
    raw.rename_channels(cname_map, allow_duplicates=False)
    raw.set_channel_types(type_map)
    raw.pick_types(eeg=True, misc=False)
    print('Electrode mapping')

    return raw, montage


def rescale_microvolts_to_volt(raw) -> mne.io.BaseRaw:
    assert np.mean(raw._data[:1000,:]) != 0, "signal is Flat (sum to zero signal)"

    sig_var = np.var(raw._data)

    if sig_var > 1:
        raw._data = raw._data * 1.e-6
    print('Rescaled signal to Volt (mean variance={sig_var})')
    return raw

def detect_artifactual_channels(raw: mne.io.BaseRaw,
                                notch_hz: int = 50):
    # Using 50Hz power variance
    psd, freqs = mne.time_frequency.psd_welch(raw, verbose=True)
    power_50hz = psd[:, np.where(freqs == notch_hz)]
    ch_art_line_var = mne.preprocessing.bads._find_outliers(power_50hz.squeeze(), threshold=3, max_iter=5, tail=0)
    print(f'{notch_hz}Hz variance rejection: {ch_art_line_var}')

    # using variance
    ch_var = [np.var(raw._data[i, :]) for i in list(range(raw._data.shape[0]))]
    ch_art_var = mne.preprocessing.bads._find_outliers(ch_var, threshold=3, max_iter=5, tail=0)
    print(f'Variance based rejection: {ch_art_var}')

    return list(set(ch_art_line_var + ch_art_var))

def apply_infinite_reference(raw: mne.io.BaseRaw, display_plot=False) -> mne.io.BaseRaw:

    raw.del_proj()  # remove our average reference projector first
    sphere = mne.make_sphere_model('auto', 'auto', raw.info)
    src = mne.setup_volume_source_space(sphere=sphere, exclude=30., pos=15.)
    forward = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
    raw_rest = raw.copy().set_eeg_reference('REST', forward=forward)

    #raw_short = ep_plot = mne.make_fixed_length_epochs(raw, duration=10)[0]
    #raw_rest_short = mne.make_fixed_length_epochs(raw_rest, duration=10)[0]

    if display_plot:
        for title, _raw in zip(['Original', 'REST (∞)'], [raw, raw_rest]):
            fig = plot_seconds(raw=_raw, seconds=10, title=title)

            #fig = raw_short.plot(n_channels=raw_short.get_data().shape[1], scalings=dict(eeg=5e-5))
            # make room for title
            fig.subplots_adjust(top=0.9)
            fig.suptitle('{} reference'.format(title), size='xx-large', weight='bold')

    return raw_rest


#Apply a variance based channel rejection if artifacts are present >30% of the time
def detec_rej_channel(raw, threshold_eeg: float,
                      reject_ratio: float,
                      window_sec=.2,
                      overlap_sec=.1,
                      show_plot=False):
    epochs_rej = mne.make_fixed_length_epochs(raw, duration=window_sec, overlap=overlap_sec, preload=True)
    epochs_rej._data.shape
    diff = np.max(epochs_rej._data, axis=2) - np.min(epochs_rej._data, axis=2)

    print(diff.shape)

    rej = (diff >= threshold_eeg).astype(np.float64)
    rel = sns.heatmap(rej)
    rel.set(title='Detected artifacts per electrode and runs (white)')

    # calculate ratio of rejected trials
    ratios = np.sum(rej, axis=0) / rej.shape[0]

    ret = np.argwhere(ratios >= reject_ratio).tolist()
    if len(ret) > 0 and len(ret[0]):
        print('Found {} channels with at least {}% {}s epochs > {} amplitude)'.format(len(ret),
                                                                                      reject_ratio * 100, window_sec,
                                                                                      threshold_eeg))
        return ret[0]
    else:
        return None


def flag_channels_as_bad(bad_channels: list, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    new_bads = [raw.info['ch_names'][ch] for ch in bad_channels]
    raw.info['bads'].extend(new_bads)
    raw.pick_types(eeg=True)


def plot_seconds(raw: mne.io.BaseRaw, seconds: float, title: str = None):
    fig = mne.make_fixed_length_epochs(raw, duration=seconds)[0].plot()
    if title is not None:
        fig.suptitle(title,  size='xx-large', weight='bold')

    return fig


def train_asr(raw: mne.io.BaseRaw) -> ASR:
    fs = int(raw.info["sfreq"])  # sampling frequency
    method = 'riemann'  # if error, use 'euclid' (still not supported out of the box 04.07.2021!)
    method = 'euclid'  # pymanopt library still buggy
    window_s = .5  # .5 sec window of analysis
    data_interval_s = None  # (begin, end) in sec of the training sample
    estimator = 'lwf'  #leave blank if using euclidian mode

    # define the ASR model using riemannian method
    #asr_model = ASR(sfreq=fs, method=method, win_len=window_s, estimator=estimator)

    # if failing (after trying twice. SVD error occurs for no reason sometimes)
    asr_model = ASR(sfreq=fs, method=method, win_len=window_s)

    # The best would be to choose another recording during the same session to train the model without overfitting
    data = raw._data  # the numpy array with data is stored in the _data variable

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
    clean = asr_model.transform(raw._data)

    display_window_s = 60  #
    fs = np.int(raw.info['sfreq'])
    if display_plot:  #
        data_p = raw._data[0:fs * display_window_s]  # reshape to (n_chans, n_times)
        clean_p = clean[0:fs * display_window_s]

        ###############################################################################
        # Plot the results
        # -----------------------------------------------------------------------------
        #
        # Data was trained on a 40s window from 5s to 45s onwards (gray filled area).
        # The algorithm then removes portions of this data with high amplitude
        # artifacts before running the calibration (hatched area = good).
        nb_ch_disp = len(raw.info['ch_names'])
        times = np.arange(data_p.shape[-1]) / fs
        f, ax = plt.subplots(nb_ch_disp, sharex=True, figsize=(32, 16))
        for i in range(nb_ch_disp):
            # ax[i].fill_between(train_idx / fs, 0, 1, color='grey', alpha=.3,
            #                   transform=ax[i].get_xaxis_transform(),
            #                   label='calibration window')
            # ax[i].fill_between(train_idx / fs, 0, 1, where=sample_mask.flat,
            #                   transform=ax[i].get_xaxis_transform(),
            #                   facecolor='none', hatch='...', edgecolor='k',
            #                   label='selected window')
            ax[i].plot(times, data_p[i], lw=.5, label='before ASR')
            ax[i].plot(times, clean_p[i], label='after ASR', lw=.5)
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

def parse_annotations(annotations:mne.annotations.Annotations,
                      acquisition_software: str,
                      speller_info: SpellerInfo,
                      stimulus_code_begin=100):


    if acquisition_software == "openvibe":
        new_annotations = openvibe.translate_annotations(annotations=annotations,
                                                         nb_rows=speller_info.nb_stimulus_rows,
                                                         nb_columns=speller_info.nb_stimulus_cols,
                                                         begin_stimuli_code=stimulus_code_begin)
    elif acquisition_software == "bci2000":
        pass
    else:
        print(f"Unknown acquisition software {acquisition_software}")

    an = new_annotations

    # Filter out string annotations from the available annotations
    string_annotations = np.where(~np.apply_along_axis(np.char.isnumeric, 0, an.description))[0]
    if string_annotations.size > 0:
        print('removing all string annotations: {}'.format(an.description[string_annotations]))
        an.delete(string_annotations)
    # Extractz row and column labels from stimuli numbers
    stimuli_labels = np.sort(np.unique(
        an.description[np.where(an.description.astype(np.uint8) >= stimulus_code_begin)]))

    row_labels = stimuli_labels[:speller_info.nb_stimulus_rows]
    col_labels = stimuli_labels[speller_info.nb_stimulus_rows:
                                speller_info.nb_stimulus_rows + speller_info.nb_stimulus_cols]

    print("Row labels: {}".format(row_labels))
    print("Col labels: {}".format(col_labels))

    # map the stimuli for MNE to read
    map_stimuli = dict(zip(stimuli_labels, stimuli_labels.astype(np.uint8)))

    #target_map = dict()
    target_map = TARGET_MAP.copy()
    target_map.update(map_stimuli)
    target_map

    return new_annotations, target_map

def metadata_from_events(events: List,
                         speller_info: SpellerInfo,
                         stimulus_code_begin: int,
                         stimulus_code_end: int = None) -> pd.DataFrame:
    """
    Create a metadata dataframe that annotate events with trial number, (padded) stimulation number and row/col information
    Ignores any event below the stimulus_code_begin
    :param events:
    :param speller_info:
    :param stimulus_code_begin:
    :return:
    """
    # automatically define the range of stimuli to be entered in the metadata
    if stimulus_code_end is None:
        stimulus_code_end = stimulus_code_begin + speller_info.nb_stimulus_rows + speller_info.nb_stimulus_cols

    # create the empty dataframe
    nb_events = np.count_nonzero(np.where(events[:, 2] <= 1))
    df_meta = pd.DataFrame(data=None, index=list(range(0, nb_events)),
                           columns=['Trial_nb', 'stim', 'is_row', 'is_col', 'is_target'])
    ev = events.copy()
    # populate with trial information
    g_trial = np.split(events[:, 2], np.where(events[:, 2] == 10)[0])[1:]

    # iterate over trials to extract their metadata
    cur_trial_n = 0
    cursor = 0
    for g in g_trial:
        cur_trial_n += 1
        #print('Trial {}'.format(cur_trial_n))
        idx_stim_labs = np.where(g >= stimulus_code_begin)
        stim_labels = g[idx_stim_labs]  # extract stimulus labels
        # isolate target and non_target stimuli
        targets_nontargets_phase = np.delete(g.copy(), idx_stim_labs)
        targets_nontargets = np.delete(targets_nontargets_phase, np.where(targets_nontargets_phase == 10))
        # extract the stimulus row/column information
        is_row = stim_labels <= speller_info.nb_stimulus_rows + stimulus_code_begin

        # deduct the cursor over stimulus event
        end_cursor = cursor + len(targets_nontargets)

        # build up the metadata dataframe
        df_meta.loc[list(range(cursor, end_cursor)), 'Trial_nb'] = cur_trial_n
        df_meta.loc[list(range(cursor, end_cursor)), 'stim'] = stim_labels
        df_meta.loc[list(range(cursor, end_cursor)), 'is_row'] = is_row.astype(np.uint8)
        df_meta.loc[list(range(cursor, end_cursor)), 'is_col'] = np.invert(is_row).astype(np.uint8)
        df_meta.loc[list(range(cursor, end_cursor)), 'is_target'] = targets_nontargets.astype(np.uint8)

        cursor = end_cursor

        assert df_meta.shape[0] == events[np.where(events[:, 2] <= 1)].shape[0],\
            f'Mismatch between number of stimuli n={df_meta.shape[0] } and their target non-target ' \
            f'description in generated metadata n={df_meta.shape[0]}'

    return df_meta

def _get_avg_target_nt(epochs: mne.Epochs):
    l_nt = epochs['NonTarget'].average()
    l_target = epochs['Target'].average()
    return l_target, l_nt

def plot_butterfly(epochs: mne.Epochs):
    l_target, l_nt = _get_avg_target_nt(epochs=epochs)

    fig, ax = plt.subplots(2, 1)
    ax1 = l_target.plot(spatial_colors=True, axes=ax[0], show=False)
    ax2 = l_nt.plot(spatial_colors=True, axes=ax[1], show=False)
    # Add title
    fig.suptitle("Target(top) - Non-Target(bottom)")
    # Fix font spacing
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return fig, l_target, l_nt

def plot_topomap(epochs: mne.Epochs):
    l_target, l_nt = _get_avg_target_nt(epochs=epochs)
    spec_kw = dict(width_ratios=[1, 1, 1, .15], wspace=0.5,
                   hspace=0.5, height_ratios=[1, 1])
    # hspace=0.5, height_ratios=[1, 2])

    fig, ax = plt.subplots(2, 4, gridspec_kw=spec_kw)
    l_target.plot_topomap(times=[0, 0.18, 0.4], average=0.05, axes=ax[0, :], show=False)
    l_nt.plot_topomap(times=[0, 0.18, 0.4], average=0.05, axes=ax[1, :], show=False)
    fig.suptitle("Target(top) - Non-Target(bottom)")
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return fig

def plot_butterfly_topomap(epochs: mne.Epochs):
    l_target, l_nt = _get_avg_target_nt(epochs=epochs)
    l_target.plot_joint()
    fig_target = plt.gcf().canvas.set_window_title('Target joint plot')
    l_nt.plot_joint()
    figt_nt = plt.gcf().canvas.set_window_title('Non-Target joint plot')
    return fig_target, figt_nt


def plot_average_erp(epochs: mne.Epochs, picks=None):
    l_target, l_nt = _get_avg_target_nt(epochs=epochs)
    evokeds = dict(NonTarget=l_nt,
                   Target=l_target)
    # picks = [f'eeg{n}' for n in range(10, 15)]
    fig_handle = mne.viz.plot_compare_evokeds(evokeds, picks=picks, combine='mean')
    return fig_handle


def plot_channel_average(epochs: mne.Epochs):
    nb_chans = epochs['Target']._data.shape[1]
    splt_width = int(np.floor(
        np.sqrt(1.0 * nb_chans + 2)))  # adding an extra plot with all channels combined at the end and a legend
    splt_height = splt_width if splt_width * splt_width >= nb_chans + 1 else splt_width + 1
    while splt_height * splt_width < nb_chans + 2:
        splt_height += 1
    fig, axes = plt.subplots(splt_height, splt_width, figsize=(10, 8), sharex=True, sharey=True)

    # evokeds = dict(NonTarget=list(epochs['NonTarget'].iter_evoked()),
    #               Target=list(epochs['Target'].iter_evoked()))
    evokeds = dict(NonTarget=epochs['NonTarget'].average(),
                   Target=epochs['Target'].average())
    # picks = [f'eeg{n}' for n in range(10, 15)]

    shape_epochs = epochs['Target']._data.shape
    nb_cells = splt_height * splt_width
    for plot_idx in range(nb_cells):

        # cells containing data
        if plot_idx in range(nb_chans):
            ch_idx = plot_idx
            print('plotting channel {}'.format(ch_idx + 1))
            mne.viz.plot_compare_evokeds(evokeds, picks=[epochs.info['ch_names'][ch_idx]],
                                         legend=False,
                                         axes=axes[plot_idx // splt_width, plot_idx % splt_width], show=False)
            # plt.show(block=False)
            plt.subplots_adjust(hspace=0.5, wspace=.5)
            # plt.pause(.05)

        # filler and legend cells
        elif plot_idx <= nb_cells - 2:
            ax = axes[plot_idx // splt_width, plot_idx % splt_width]
            ax.clear()  # clears the random data I plotted previously
            ax.set_axis_off()  # removes the XY axes

            if plot_idx == nb_cells - 2:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                leg = ax.legend(handles, labels)

    print('plotting averaged channels')
    fig = mne.viz.plot_compare_evokeds(evokeds, picks=epochs.info['ch_names'], combine='mean',
                                     legend=False,
                                     axes=axes[-1, -1], show=False)

    # retrieve the legend and move it in the previous cell

    plt.subplots_adjust(hspace=1, wspace=.1)
    plt.show()
    return fig

def plot_erp_heatmaps(epochs: mne.Epochs):
    epochs['Target'].plot_image(combine='mean')
    fig_t = plt.gcf().canvas.set_window_title('Target')
    epochs['NonTarget'].plot_image(combine='mean')
    fig_nt = plt.gcf().canvas.set_window_title('Non-Target')

    return fig_t, fig_nt

def plot_erp_heatmaps_channelwise(epochs: mne.Epochs, csd_applied: bool):
    dict_electrodes = dict(eeg='EEG') if not csd_applied else dict(csd='CSD')
    for ch_type, title in dict_electrodes.items():
        layout = mne.channels.find_layout(epochs.info, ch_type=ch_type)
        fig_t = epochs['Target'].plot_topo_image(layout=layout, fig_facecolor='w',
                                         font_color='k', title=title + ' Target Trial x time amplitude')
        fig_nt = epochs['NonTarget'].plot_topo_image(layout=layout, fig_facecolor='w',
                                            font_color='k', title=title + ' Non-Target Trial x time amplitude')

        return fig_t, fig_nt

def _reshape_mne_raw_for_lda(X: np.ndarray, verbose=False) -> np.ndarray:
    """
    reshapes raw data for LDA, flattening the last dimension
    """
    if verbose:
        print('Data shape from MNE {}'.format(X.shape))
    X_out = np.moveaxis(X, 1, -1)
    if verbose:
        print('new data shape with sampling prioritized over channels {}'.format(X_out.shape))
    X_out = X_out.reshape([X_out.shape[0], X_out.shape[1] * X_out.shape[2]], order='C')
    if verbose:
        print('Shape for K-fold LDA {}'.format(X_out.shape))
    return X_out

def conf_matrix(y: np.ndarray, pred: np.ndarray):
    cmat = metrics.confusion_matrix(y, pred)
    cmat_norm = metrics.confusion_matrix(y, pred,
                                         normalize='true')
    ((tn, fp), (fn, tp)) = cmat
    ((tnr, fpr), (fnr, tpr)) = cmat_norm

    fig = plt.figure()
    labels = ['Non-Target', 'Target']
    sns.heatmap(cmat, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')

    # alternative using sklearn plots
    # plt.figure()
    # from sklearn.metrics import ConfusionMatrixDisplay
    # cm_display = ConfusionMatrixDisplay(cmat).plot()

    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})',
                          f'FP = {fp} (FPR = {fpr:1.2%})'],
                         [f'FN = {fn} (FNR = {fnr:1.2%})',
                          f'TP = {tp} (TPR = {tpr:1.2%})']],
                        index=['True 0(Non-Target)', 'True 1(Target)'],
                        columns=['Pred 0(Non-Target)',
                                 'Pred 1(Target)']), fig

def run_single_epoch_LDA_analysis(X_data: np.ndarray,
                                  y_true_labels: np.ndarray,
                                  nb_k_fold: int,
                                  display_confidence_matrix: bool = True,
                                  display_precision_recall: bool = True):
    # Reshaping data
    X = _reshape_mne_raw_for_lda(X_data)
    y = y_true_labels
    # Separating train and test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    kf = KFold(n_splits=nb_k_fold)
    kf.get_n_splits(X)

    list_score = []
    list_auc = []

    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X[train_index], X[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]
        clf.fit(X_train_kf, y_train_kf)
        kscore = clf.score(X_test_kf, y_test_kf)
        y_pred_kf = clf.predict(X_test_kf)
        k_auc = roc_auc_score(y_test_kf, y_pred_kf)
        print('fold score: {}, AUC={}'.format(np.round(kscore, decimals=3), np.round(k_auc, decimals=3)))
        list_score.append(kscore)
        list_auc.append(k_auc)

    print('Average score {}-Fold = {}, AUC={}'.format(kf.get_n_splits(X), np.round(np.mean(list_score), decimals=2),
                                                      np.round(np.mean(list_auc), decimals=2)))

    # using the training/validate samples,
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    print('Score training-validation {}, AUC={}'.format(np.round(score, decimals=2), np.round(auc, decimals=2)))

    # compare the count of targets to the amount of stimuli
    target_to_nontarget_ratio = np.sum(y) / y.shape[0]
    if target_to_nontarget_ratio != .5:
        print(f'Score is only valid if classes are balanced target_ratio={target_to_nontarget_ratio:0.2f}\n'
              f' please refer to AUC instead')

    if display_confidence_matrix:
        df_confidence, fig_confidence = conf_matrix(y=y_test, pred=y_pred)

    y_score = clf.decision_function(X_test)
    if display_precision_recall:
        fig_roc = plot_precision_recall(classifier=clf, X=X_test, y_gt=y_test)
    else:
        fig_roc = None

    return fig_confidence, fig_roc


def plot_precision_recall(classifier: sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
                          X: np.ndarray,
                          y_gt: np.ndarray):

    y_score = classifier.decision_function(X)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_gt, y_score, pos_label=classifier.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)  # .plot()

    # Precision Recall Display
    prec, recall, _ = precision_recall_curve(y_gt, y_score,
                                             pos_label=classifier.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)  # .plot()

    # Display them side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    roc_display.plot(ax=ax1)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    ax1.legend(loc="lower right")
    pr_display.plot(ax=ax2)

    plt.show()
    return fig

def signed_r_square(epochs: mne.Epochs, time_epoch: Tuple[float], display_rsq_plot: bool = True):
    rsq = signed_r_square_mne(epochs, classes=['Target', 'NonTarget'])
    # make a pandas database to properly display electrodes and samples
    fs = epochs.info['sfreq']
    x = np.float64(list(range(rsq.shape[1]))) * (1000 / fs)
    x = x.round(decimals=0).astype(np.int64) + np.int64(time_epoch[0] * 1000)
    df_rsq = pd.DataFrame(rsq, columns=x, index=epochs.info['ch_names'])

    fig_rsq = plt.figure()
    ax = sns.heatmap(df_rsq, linewidths=0, cmap="coolwarm").set(title='Signed r-square maps Target vs Non-Target',
                                                                xlabel='Time (ms)')

    return rsq, fig_rsq




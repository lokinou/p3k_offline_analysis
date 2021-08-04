from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import os
import re
import mne.io
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from meegkit.asr import ASR

from p3k import openvibe
from p3k import bci2000

@dataclass
class SpellerInfo:
    nb_stimulus_rows: int = None
    nb_stimulus_cols: int = None
    nb_seq: int = None

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
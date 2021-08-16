from BCI2kReader import BCI2kReader as b2k
from BCI2kReader import FileReader as f2k

import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

raws = []


def extract_annotations(filename, begin_stimuli_code: int, verbose=False):
    display_preprocessing_plots = False
    file = b2k.BCI2kReader(filename)

    if verbose:
        print(file.states)
    target_states = np.squeeze(file.states['StimulusType'])
    stimulus_codes = np.squeeze(file.states['StimulusCode'])
    if 'StimulusBegin' in file.states.keys():
        stimulus_begin = np.squeeze(file.states['StimulusBegin'])
    else:
        stimulus_begin = np.squeeze(file.states['Flashing'])

    phase = np.squeeze(file.states['PhaseInSequence'])

    fs = file.samplingrate

    idx_targets = np.where(target_states)[0]
    idx_codes = np.where(stimulus_codes > 0)[0]
    idx_begin = np.where(stimulus_begin > 0)[0]

    # In BCI2000 states are maintained over different samples, we search here the differences of when the codes are > 0
    groups = np.split(idx_codes, np.where(np.diff(idx_codes) != 1)[0] + 1)
    # we take the first sample where a difference can be found
    code_change_idx = np.array([g[0] for g in groups])
    # [idx_codes[idx] for idx in code_change_idx]
    print('nb stimuli={}'.format(len(code_change_idx)))

    # we intersect the target index list with the code change to find the onset of targets and non-targets
    target_idx = np.intersect1d(code_change_idx, idx_targets)
    print('nb targets={}'.format(len(target_idx)))
    non_target_idx = np.setdiff1d(code_change_idx, idx_targets)

    # Translating into MNE Annotations
    # define the annotations from the recovered stimuli (in seconds)
    sample_lengh = 1 / fs
    onsets = code_change_idx * sample_lengh
    onsets = np.repeat(onsets, 2)  # repeat onsets
    # define the descriptio
    description_targets = np.zeros(code_change_idx.shape, dtype=np.uint)
    # index of targets in the list of stimuli onsets
    description_targets[np.searchsorted(code_change_idx, target_idx)] = 1
    description_codes = stimulus_codes[
                            code_change_idx] + begin_stimuli_code  # start codes at 100 because 0 and 1 are used for target and nontarget
    # merge code and target decriptions
    description = np.zeros(description_targets.shape[0] * 2, dtype=np.uint)
    description[np.arange(description_targets.shape[0] * 2, step=2)] = description_codes
    description[np.arange(start=1, stop=(description_targets.shape[0] * 2) + 1, step=2)] = description_targets

    if display_preprocessing_plots:
        fig = plt.figure()
        plt.plot(description[:100])
        fig.suptitle('Targets(1) and non-targets(0) for 100 first stimuli')

    if display_preprocessing_plots:
        fig = plt.figure()
        plt.plot(phase == 1)
        fig.suptitle('Trial begin')

    # extract trial begin markers  #  this method does not work since some stimuli are declared before phase==1
    # let's think baclwards use the end markers instead
    new_phase_continuous = np.where(phase == 1)[0]
    groups = np.split(new_phase_continuous, np.where(np.diff(new_phase_continuous) != 1)[0] + 1)
    new_trial_idx = np.array([g[0] for g in groups]) if len(groups) > 1 else None

    # extract trial end markers
    new_phase_continuous = np.where(phase == 3)[0]
    groups = np.split(new_phase_continuous, np.where(np.diff(new_phase_continuous) != 1)[0] + 1)
    end_of_trial_idx = np.array([g[-1] for g in groups])  # take the last index to integrate all post sequence duration

    # deduce trial begin markers  #
    # new_trial_idx = np.zeros(end_of_trial_idx.size)
    # new_trial_idx[1:] = end_of_trial_idx[1:]+1

    print(new_trial_idx)
    print(end_of_trial_idx)

    if new_trial_idx is None:
        print(
            'WARNING: markers for begin trial (PhaseInSequence=1) missing (in old brain painting dev versions)!!!, using end of trial instead')
        new_trial_idx = [0]
        new_trial_idx.extend(end_of_trial_idx[0:-1])  # deduce the bounds from end of trial
        new_trial_idx = np.array(new_trial_idx)  # convert to numpy
        print(new_trial_idx)

    if new_trial_idx.shape[0] > end_of_trial_idx.shape[0]:
        print(
            'WARNING: no end of trial for the last trial (interrupted recording?), it will be ignored for offline accuracy calculation')
        inter_trial_duration = end_of_trial_idx[0:len(new_trial_idx)] - new_trial_idx
    else:
        inter_trial_duration = end_of_trial_idx - new_trial_idx

    inter_trial_duration = inter_trial_duration * sample_lengh  # express in seconds

    print("Extracted {} trials".format(len(new_trial_idx)))

    # set a non-zero duration for stimuli (or MNE ignores them)
    duration = np.ones(onsets.shape) * sample_lengh

    # merge phase in sequence events with stimuli onsets
    onsets_phase = new_trial_idx * sample_lengh
    onsets = np.concatenate((onsets, onsets_phase))

    duration = np.concatenate((duration, inter_trial_duration))
    description = np.concatenate((description, np.ones(new_trial_idx.shape) * 10))  # concatenate trials markers=10
    srt = np.argsort(onsets)  # sort according to their timing
    onsets = onsets[srt]
    duration = duration[srt]
    description = description[srt].astype(np.uint8)
    inter_trial_duration
    annotations = mne.Annotations(onset=onsets, duration=duration, description=description)

    file.flush()
    return annotations


def load_bci2k(filename_list, begin_stimuli_code: int = 0, verbose=False) -> Tuple[List[mne.io.RawArray], Tuple[int]]:
    """
    return MNE raw, number of rows in the matrix
    :param filename_list: list of filenames to load and concatenate
    :param begin_stimuli_code: Integer added to the stimuli code
    :param verbose:
    :return: (list of mne raw arrays, (number of rows, columns sequences))
    """
    raws = []
    for fn in filename_list:
        cname = None
        with b2k.BCI2kReader(fn) as file:

            # Extract signals and states
            print('Reading {}'.format(fn))
            eeg_data = file.signals
            states = file.states
            fs = file.samplingrate
            nb_chan = eeg_data.shape[0]
            # file.purge()

            # Extract channel names
            reader = f2k.bcistream(fn)
            if verbose:
                print(reader.params)
            # actualize the parameters by including the defined channel names
            if len(reader.params['ChannelNames']):
                if cname != reader.params['ChannelNames']:
                    cname = reader.params['ChannelNames']
                    print('Actualized channel names to {}'.format(cname))

            if cname is None:
                cname = [str(ch_n) for ch_n in list(range(nb_chan))]

            # extract the number of rows
            nb_stim_rows = np.uint8(reader.params['NumMatrixRows'][0])
            nb_stim_cols = np.uint8(reader.params['NumMatrixColumns'][0])
            nb_seq = np.uint8(reader.params['NumberOfSequences'])

            # convert states into annotations
            info = mne.create_info(cname, fs, ch_types='eeg', verbose=None)
            raw_array = mne.io.RawArray(eeg_data, info)
            # Manually force the filename or mne complains
            raw_array._filenames = [os.path.basename(fn)]

            annotations = extract_annotations(fn, begin_stimuli_code=begin_stimuli_code, verbose=False)
            raw_array.set_annotations(annotations)
            raws.append(raw_array)
    return raws, (nb_stim_rows, nb_stim_cols, nb_seq)

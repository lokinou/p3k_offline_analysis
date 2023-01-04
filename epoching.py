
import numpy as np
import pandas as pd
import mne
from typing import List

from p3k.params import SpellerInfo, InternalParameters
from p3k.read.bci_format import openvibe


def parse_annotations(annotations:mne.annotations.Annotations,
                      acquisition_software: str,
                      speller_info: SpellerInfo,
                      internal_params: InternalParameters):


    if acquisition_software == "openvibe":
        new_annotations = openvibe.translate_annotations(annotations=annotations,
                                                         nb_rows=speller_info.nb_stimulus_rows,
                                                         nb_columns=speller_info.nb_stimulus_cols,
                                                         begin_stimuli_code=internal_params.STIMULUS_CODE_BEGIN)
    elif acquisition_software == "bci2000":
        new_annotations = annotations.copy()
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
        an.description[np.where(an.description.astype(np.uint8) >= internal_params.STIMULUS_CODE_BEGIN)]))

    row_labels = stimuli_labels[:speller_info.nb_stimulus_rows]
    col_labels = stimuli_labels[speller_info.nb_stimulus_rows:
                                speller_info.nb_stimulus_rows + speller_info.nb_stimulus_cols]

    print("Row labels: {}".format(row_labels))
    print("Col labels: {}".format(col_labels))

    # map the stimuli for MNE to read
    map_stimuli = dict(zip(stimuli_labels, stimuli_labels.astype(np.uint8)))

    target_map = internal_params.TARGET_MAP.copy()
    target_map.update(map_stimuli)

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

def get_avg_target_nt(epochs: mne.Epochs):
    l_nt = epochs['NonTarget'].average()
    l_target = epochs['Target'].average()
    return l_target, l_nt



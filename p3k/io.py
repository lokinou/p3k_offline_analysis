import os
from typing import Tuple

import mne

from p3k.p3k import SpellerInfo
from p3k import bci2000, openvibe


def load_eeg_from_folder(data_path: str,
                         speller_info: SpellerInfo,
                         begin_stimuli_code: int,
                         fix_openvibe_annotations: bool = True) -> Tuple[mne.io.BaseRaw, str, SpellerInfo]:
    nb_stimlus_rows = None  # stores the number of rows in the P300 to separate rows and columns
    os.path.exists(data_path)
    fnames = []
    for file in os.listdir(data_path):
        if file.endswith(".gdf"):
            acquisition_software = 'openvibe'
            fnames.append(os.path.join(data_path, file))
            print(os.path.join(data_path, file))
        elif file.endswith(".dat"):
            acquisition_software = 'bci2000'
            print(os.path.join(data_path, file))
            fnames.append(os.path.join(data_path, file))
    print(fnames)
    if acquisition_software == 'openvibe':
        # load and preprocess data ####################################################
        raws = [mne.io.read_raw_gdf(f, preload=True) for f in fnames]
        #nb_stimlus_rows, nb_stimulus_cols, nb_seq = ov_nb_row_stims, ov_nb_col_stims, ov_nb_sequences
        print(f"Using user defined SpellerInfo from files {SpellerInfo}")
        if fix_openvibe_annotations:
            for r in raws:
                new_annotations = openvibe.fix_openvibe_annotations(r.annotations)
                r.set_annotations(new_annotations)
        raw = mne.concatenate_raws(raws)


    elif acquisition_software == 'bci2000':
        raws, (nb_stimulus_rows, nb_stimulus_cols, nb_seq) = bci2000.load_bci2k(fnames,
                                                                                begin_stimuli_code=begin_stimuli_code,
                                                                                verbose=False)
        SpellerInfo.nb_stimulus_rows = nb_stimulus_rows
        SpellerInfo.nb_stimulus_cols = nb_stimulus_cols
        SpellerInfo.nb_seq = nb_seq
        print(f"Actualized SpellerInfo from files {SpellerInfo}")
        raw = mne.concatenate_raws(raws)



    return (raw, acquisition_software, speller_info)


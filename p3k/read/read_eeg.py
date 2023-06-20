import os
import mne
import numpy as np
from typing import Tuple, Union
from pathlib import Path

from p3k.params import SpellerInfo
from p3k.read.bci_format import openvibe, bci2000

def load_eeg_from_files(eeg_files: Union[str, list, Path],
                        speller_info: SpellerInfo,
                        begin_stimuli_code: int,
                        rescale_to_volt: bool = True,
                        fix_openvibe_annotations: bool = True) -> Tuple[mne.io.BaseRaw, str, SpellerInfo]:
    nb_stimlus_rows = None  # stores the number of rows in the P300 to separate rows and columns
    if isinstance(eeg_files, str) or isinstance(eeg_files, Path):
        input_files = [eeg_files]
    elif isinstance(eeg_files, list):
        input_files = eeg_files
    else:
        raise TypeError(f'Unknown type for eeg_files: {type(eeg_files)}')

    acquisition_software = None
    files_mismatch = False
    for file in eeg_files:
        if file.suffix.lower() == ".gdf":
            files_mismatch = acquisition_software == bci2000
            acquisition_software = 'openvibe'
        elif file.suffix.lower() == ".dat":
            files_mismatch = acquisition_software == 'openvibe'
            acquisition_software = 'bci2000'
    if files_mismatch:
        raise AssertionError(f"Can't load both openvibe and bci2000 at once")

    if acquisition_software == 'openvibe':
        # load and preprocess data ####################################################
        raws = [mne.io.read_raw_gdf(f, preload=True) for f in eeg_files]
        #nb_stimlus_rows, nb_stimulus_cols, nb_seq = ov_nb_row_stims, ov_nb_col_stims, ov_nb_sequences
        print(f"Using user defined SpellerInfo from files {SpellerInfo}")
        if fix_openvibe_annotations:
            for r in raws:
                new_annotations = openvibe.fix_openvibe_annotations(r.annotations)
                r.set_annotations(new_annotations)
        raw = mne.concatenate_raws(raws)


    elif acquisition_software == 'bci2000':
        raws, (nb_stimulus_rows, nb_stimulus_cols, nb_seq) = bci2000.load_bci2k(eeg_files,
                                                                                begin_stimuli_code=begin_stimuli_code,
                                                                                verbose=False)
        speller_info.nb_stimulus_rows = nb_stimulus_rows
        speller_info.nb_stimulus_cols = nb_stimulus_cols
        speller_info.nb_seq = nb_seq
        print(f"Actualized SpellerInfo from files {speller_info}")
        raw = mne.concatenate_raws(raws)

        if rescale_to_volt:
            raw = _rescale_microvolts_to_volt(raw)


    return (raw, acquisition_software, speller_info)


def _rescale_microvolts_to_volt(raw) -> mne.io.BaseRaw:
    assert np.mean(raw._data[:1000,:]) != 0, "signal is Flat (sum to zero signal)"

    sig_var = np.var(raw._data)

    if sig_var > 1:
        raw._data = raw._data * 1.e-6
    print('Rescaled signal to Volt (mean variance={sig_var})')
    return raw
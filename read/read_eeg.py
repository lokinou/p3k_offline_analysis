import os
import mne
import numpy as np
from typing import Tuple

from p3k.params import SpellerInfo
from p3k.read.bci_format import openvibe, bci2000

def load_eeg_from_folder(data_path: str,
                         speller_info: SpellerInfo,
                         begin_stimuli_code: int,
                         rescale_to_volt: bool = True,
                         fix_openvibe_annotations: bool = True) -> Tuple[mne.io.BaseRaw, str, SpellerInfo]:
    nb_stimlus_rows = None  # stores the number of rows in the P300 to separate rows and columns
    assert os.path.exists(data_path), f"The folder containing data does not exist {data_path}"
    fnames = []
    print(f"Current path. {os.path.abspath(os.curdir)}")
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
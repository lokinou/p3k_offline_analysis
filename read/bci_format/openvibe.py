import mne.io
import mne
import numpy as np
import pandas as pd
import re

OV_STIMULUS_BEGIN = 33025
#These annotations seem to relate to hex codes. OpenViBE definitions
# can be found on [OpenViBE's website](http://openvibe.inria.fr/stimulation-codes/). Let's parse the copypasted list
MAP_REPLACE_STIM = {'33286': 0, '33285': 1, '32773': 10}


def fix_openvibe_annotations(annotations: mne.annotations.Annotations,
                             verbose: bool = False) -> mne.annotations.Annotations:
    """
    Openvibe conversion to GDF introduces a wrong label 0 or 1 and shifts the list of stimuli
    This function shift the stimuli back, although the very last stimulus is lost in the process.
    :param annotations:
    :param verbose:
    :return:
    """
    annotations = annotations.copy()

    print(f"Erroneous annotations: {annotations.description}")
    annotations.description = np.roll(annotations.description, -1)
    annotations.delete(-1)
    print(f"Corrected annotations: {annotations.description}")

    # in case you want to debug the issue, I left here a way to visualize them
    # retrieving the list of annotations
    import pprint
    if verbose:
        df = annotations.to_data_frame()
        print('Displaying all annotations')
        annot_codes = [np.int64(n) for n in np.unique(df['description'])]
        df['description'] = df['description'].astype(int)

        # to see and debug the fill list of annotations

        pd.set_option('display.max_rows', None)
        # a = df[df['description'] != 33286]
        # print(a)
        print(df)
        pd.set_option('display.max_rows', 32)

    return annotations


def translate_annotations(annotations: mne.annotations.Annotations,
                          nb_rows: int, nb_columns: int,
                          begin_stimuli_code: int):
    annotations = annotations.copy()

    map_stim = MAP_REPLACE_STIM.copy()
    # replace openvibe stimuli by human readable ones


    stim_ov_list = np.arange(OV_STIMULUS_BEGIN, OV_STIMULUS_BEGIN + nb_rows + nb_columns).astype(str)
    stim_list = np.arange(nb_rows + nb_columns) + 1 + begin_stimuli_code  # add 100 to stimuli

    map_stim.update(dict(zip(stim_ov_list, stim_list)))

    ov_nb_sequences = 10
    ov_nb_row_stims = 7
    ov_nb_col_stims = 7
    # delete segments that contain '0' or '1' stimuli
    annotations.delete(np.where(annotations.description == '0'))
    annotations.delete(np.where(annotations.description == '1'))
    annotations.delete(
        np.where(annotations.description.astype(np.uint8) > 1000)[0])  # remove other stimulations

    # remap stimuli
    annotations.description = pd.DataFrame(annotations.description).replace(
        map_stim).to_numpy().squeeze().astype(str)

    tr_sim = ''
    pat_extract = re.compile('^([^ ]+)[ ]+0x[0-9A-Fa-f]+[ \/]+([0-9]+)')
    # OVTK_GDF_125_Watt                                     0x585       //  1413
    k_stim = []
    k_stim_int = []
    v_stim = []

    # Making a list of the annotations to check whether all stimuli can be found

    # read and convert annotations
    with open(r'.\ov_stims.txt', 'r') as fd:
        for line in fd.readlines():
            m = pat_extract.match(line)
            v, k = m.groups()
            k_stim.append(k)
            k_stim_int.append(int(k))
            v_stim.append(v)

    # format dict and list
    stim_map = dict(zip(k_stim_int, v_stim))
    stim_map_inv = dict(zip(v_stim, k_stim))

    stim_tup = list(zip(k_stim_int, v_stim))

    df = pd.DataFrame.from_dict(stim_tup)
    df.columns = ['coden', 'desc']
    annot_codes = [np.int64(n) for n in np.unique(df['coden'])]
    df[[c in annot_codes for c in df.coden]]

    return annotations

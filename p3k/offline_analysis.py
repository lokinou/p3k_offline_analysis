import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple

@dataclass
class Parameters:
    cname: List[int] = None
    time_epoch: Tuple[float] = (-.200, .600)
    time_baseline: Tuple[float] = (-.200, 0)
    resample_freq: int = None
    reject_channels_full_of_artifacts: bool = False
    reject_artifactual_epochs: bool = False
    artifact_threshold: float = 100e-6
    ratio_tolerated_artifacts: float = 0.3

@dataclass
class ParamsLDA:
    resample_LDA: int = 32
    nb_cross_fold: int = 1

@dataclass
class ParamsCosmetic:
    display_channel_erp: List[int] = None
    export_figures: bool = True



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
    best_channel_erp: bool = True
    offline_accuracy: bool = True

TARGET_MAP = {'0':0, '1':1, '10':10}




def _make_output_folder(filename_s: Union[str, List[str]], fig_folder: str) -> str:
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


def run_analysis(input_folder: str,
                 display_plots: DisplayPlots = None,
                 output_folder: str = None):
    print("run_analysis todo")
    pass







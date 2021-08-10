from dataclasses import dataclass, field
from typing import Union, List, Tuple


@dataclass
class ParamChannels:
    cname: List[int] = None


@dataclass
class InternalParameters:
    TARGET_MAP: dict = field(init=True, default_factory=dict)
    STIMULUS_CODE_BEGIN: int = 100
    EVENT_IDS: dict = field(init=True, default_factory=dict)

    def __post_init__(self):
        self.TARGET_MAP = {'0': 0, '1': 1, '10': 10}
        self.EVENT_IDS = {'NonTarget': 0, 'Target': 1}


@dataclass
class ParamEpochs:
    time_epoch: Tuple[float] = (-.200, .600)
    time_baseline: Tuple[float] = (-.200, 0)


class ParamData:
    data_dir: str = r"./data_sample"
    acquisition_software = None  # bci2000 or openvibe or None for autodetection

@dataclass
class ParamPreprocessing:
    resample_freq: int = None
    apply_resample: bool = field(init=False, default=False)
    apply_infinite_reference = False  # re-referencing
    apply_ASR = True  # use Artifact Subspace Reconstruction (artifact removal)
    apply_CSD = False  # use Current Source Density (spatial filter)

    def __post_init__(self):
        self.apply_resample = self.resample_freq is not None

@dataclass
class ParamArtifacts:
    reject_channels_full_of_artifacts: bool = False
    reject_artifactual_epochs: bool = False
    artifact_threshold: float = 100e-6
    ratio_tolerated_artifacts: float = 0.3


@dataclass
class ParamLDA:
    resample_LDA: int = 32
    nb_cross_fold: int = 1


@dataclass
class ParamInterface:
    display_channel_erp: List[int] = None
    export_figures_path: str = 'out'
    export_figures: bool = field(init=False, default=False)

    def __post_init__(self):
        self.export_figures = self.export_figures_path is not None


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
    channel_artifacts = True
    reject_epochs: bool = True
    butterfly: bool = False
    butterfly_topomap: bool = True
    channel_average: bool = True
    erp_heatmap: bool = False
    erp_heatmap_channelwise: bool = False
    signed_r_square: bool = True
    best_channel_erp: bool = True
    offline_accuracy: bool = True

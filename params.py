from dataclasses import dataclass, field
from typing import Union, List, Tuple


@dataclass
class ParamChannels:
    cname: List[str] = None
    select_subset: List[str] = None


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


@dataclass
class ParamData:
    data_dir: str = None
    acquisition_software = None  # bci2000 or openvibe or None for autodetection


@dataclass
class ParamPreprocessing:
    resample_freq: int = None
    bandpass: tuple = (.5, 40)
    apply_notch: bool = False
    notch: int = 50
    apply_resample: bool = field(init=False, default=False)
    apply_infinite_reference: bool = True  # re-referencing
    apply_ASR: bool = False  # use Artifact Subspace Reconstruction (artifact removal)
    apply_CSD: bool = False  # use Current Source Density (spatial filter)

    def __post_init__(self):
        self.apply_resample = self.resample_freq is not None

@dataclass
class ParamArtifacts:
    reject_channels_full_of_artifacts: bool = False
    reject_artifactual_epochs: bool = False
    correct_artifacts_asr: bool = False
    artifact_threshold: float = 100e-6
    ratio_tolerated_artifacts: float = 0.3


@dataclass
class ParamLDA:
    resample_LDA: int = 64
    nb_cross_fold: int = 5


@dataclass
class ParamInterface:
    display_channel_erp: List[int] = None
    export_figures_path: str = 'results'
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
    montage_plots: bool = False
    csd: bool = False
    cross_correlation: bool = False
    epochs: bool = True
    channel_artifacts = True
    reject_epochs: bool = True
    butterfly: bool = False
    butterfly_topomap: bool = False
    channel_average: bool = True
    erp_heatmap: bool = False
    erp_heatmap_channelwise: bool = False
    signed_r_square: bool = True
    best_channel_erp: bool = True
    offline_accuracy: bool = True
    score_table: bool = True
    fixed_display_range: list = field(default_factory=list) # Matthias


@dataclass
class SampleParams:
    """
    Contains a few parameters for the sample given with the package to test it out,.
    It is called if P300Analysis.run_analysis() is called without any parameters
    """
    data_dir: str = r"./data_sample"
    channels: list = field(init=True, default_factory=list)
    speller_info: SpellerInfo = field(init=False)

    def __post_init__(self):
        self.channels = ['Fz', 'FC1', 'FC2', 'C1', 'Cz', 'C2', 'P3', 'Pz', 'P4', 'Oz']
        self.speller_info = SpellerInfo(nb_stimulus_rows=7, nb_stimulus_cols=7, nb_seq=10)

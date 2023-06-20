from dataclasses import dataclass, field
from typing import Union, List, Tuple
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


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
    data_files: list = field(init=True, default_factory=list)
    filename_filter: str = '.*'  # filename filter (or case insensitive regular expression)
    extension_filter: str = None  # extension filter (or regular expression)
    acquisition_software = None  # bci2000 or openvibe or None for autodetection

    def __post_init__(self):

        filter_pat = re.compile(self.filename_filter, re.IGNORECASE)

        # assertions
        assert (self.data_files is not None)
        if isinstance(self.data_files, str):
            self.data_files = [Path(self.data_files)]
        elif isinstance(self.data_files, Path):
            self.data_files = [self.data_files]

        assert isinstance(self.data_files, list)

        valid_files = []

        for fpath in self.data_files:
            path_eeg_file = Path(fpath) if isinstance(fpath, str) else fpath
            if not path_eeg_file.exists():
                raise FileNotFoundError(path_eeg_file.absolute())

            # if a directory is provided, iterate and find the files required
            if path_eeg_file.is_dir():
                # lookup inside the folder
                for file in path_eeg_file.iterdir():
                    if file.is_file():
                        if self.extension_filter is not None:
                            if re.match(self.extension_filter, file.suffix) \
                                or file.suffix.find(self.extension_filter) >= 0:
                                valid_files.append(file)
                                logger.debug(f"found {file}")
                        else:
                            if file.suffix.lower() == ".gdf":
                                # acquisition_software = 'openvibe'
                                valid_files.append(file)
                                logger.debug(f"found {file}")
                            elif file.suffix.lower() == ".dat":
                                # acquisition_software = 'bci2000'
                                valid_files.append(file)
                                logger.debug(f"found {file}")
            else:
                valid_files.append(path_eeg_file)

        # use regular expression to filter it
        valid_files = [vf for vf in valid_files if filter_pat.match(vf.stem) or vf.stem.find(filter_pat) >= 0]

        self.data_files = valid_files if len(valid_files) > 0 else None

@dataclass
class ParamPreprocessing:
    resample_freq: int = None
    bandpass: tuple = (.5, 40)
    notch: int = 50
    apply_resample: bool = field(init=False, default=False)
    apply_infinite_reference: bool = True  # re-referencing
    apply_ASR: bool = False  # use Artifact Subspace Reconstruction (artifact removal)
    apply_CSD: bool = False  # use Current Source Density (spatial filter)
    #apply_XDAWN: bool = False  # use XDAWN, would require to train and fit the epochs during cross validation ugh

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
    fixed_display_range: list = None # Matthias


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

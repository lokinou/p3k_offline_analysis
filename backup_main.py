
from p3k.P300Analysis import run_analysis
from pathlib import Path
from p3k.params import ParamChannels, ParamData, SpellerInfo, \
    ParamPreprocessing, ParamArtifacts, ParamEpochs, ParamInterface, DisplayPlots
import logging

#logging.basicConfig(level=logging.DEBUG)
#logging.debug('debug logging activated')
logger = logging.getLogger(__name__)

# channel
#p_channels = ParamChannels(cname=['Fz','FC1','FC2','C1','Cz','C2','P3','Pz','P4','Oz'])
# P300 speller description
#speller_info = SpellerInfo(nb_stimulus_rows=7, nb_stimulus_cols=7, nb_seq=10)


# Data location
data_folder = Path(r"C:\Users\mae08yc\Desktop\P3k Pipeline\Data")

# Specify folders:
list_folders = ["tacCalib051",#],
                "VP02_Calibration001", "VP04_Calibration001"]
list_folders = ["VP02_Calibration001"] # works
list_folders = ["VP04_Calibration001"] # error

#p_data = ParamData(data_dir=r'C:\Users\mae08yc\Desktop\P3k Pipeline\Data\tacCalib051')


electrodes = ["C4"]
#electrodes = ["Fz","Cz","Pz"]

# Epoch time window
p_epoch = ParamEpochs(time_epoch=(-.200,0.8))

# Deviations from default parameters:
p_preproc = ParamPreprocessing(apply_infinite_reference=False,
                               apply_ASR=False)

p_artifact = ParamArtifacts(reject_channels_full_of_artifacts=False,
                            reject_artifactual_epochs=False,  # Todo: classification will not work with non-divisible number of epochs
                            correct_artifacts_asr=False, # DO NOT USE, Loic somehow implemented it twice: p_artifact.correct_artifacts_asr AND p_preproc.apply_ASR
                            artifact_threshold=70e-6)  # default 100 µV

p_plots = DisplayPlots(channel_average=True, epochs=False)

run_offline_classification = False

'''
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
'''

#p_int = ParamInterface(export_figures=True)

Skip_Plot_Visualization = False
                # False: Execution halts until plots are closed manually
                # True:  Plots safed, but not shown
                # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib

# Figures
if Skip_Plot_Visualization:
    import matplotlib
    matplotlib.use('Agg')


# Loop through all folders # Todo: Return and store
for name_folder in list_folders:
    logger.info(f"Processing folder {name_folder}")
    path_folder = data_folder.joinpath(name_folder)
    p_data = ParamData(data_dir=path_folder)


    # AY LES GOOOO!
    run_analysis(param_data=p_data,
                 param_preproc=p_preproc,
                 param_artifacts=p_artifact,
                 param_epochs=p_epoch,
                 display_plots=p_plots, #, param_channels=p_channels)  # , speller_info=speller_info)
                 current_folder=name_folder,
                 electrodes=electrodes,
                 classify=run_offline_classification)



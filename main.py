
# !!! Tested only with MNE 1.2.2

from p3k.P300Analysis import run_analysis
from pathlib import Path
from p3k.params import ParamChannels, ParamData, SpellerInfo, \
    ParamPreprocessing, ParamArtifacts, ParamEpochs, ParamInterface, DisplayPlots
import logging
import pandas as pd                          # for data frame
from glob import glob                        # for folder and file listings
from numpy import squeeze
import os

#logging.basicConfig(level=logging.DEBUG)
#logging.debug('debug logging activated')
logger = logging.getLogger(__name__)


### SETUP ###

## INPUT:  Data location
data_folder = Path(r"C:\Users\mae08yc\Desktop\Test")
data_folder = Path(r"G:\Matthias_Data\STUDIES\Frankenstein\STUDY_DATA")

# Specify FOLDERS with Pattern Matching:
match = "VP_19*"
excluded = ("VP_11001", "VP_15001", "VP_17001", "VP_36001")

list_folders = glob(str(data_folder) + "\\" + match, recursive=False)
list_folders = [item.split("\\")[-1] for item in list_folders]          # take folder name only -> split path, take LAST (-1) element
list_folders = [i for i in list_folders if i not in excluded]
assert len(list_folders) > 0, "No input data found."



## OUTPUT location
extracted_out_file = os.path.join('results', 'mean_amplitudes.txt')# data_folder.joinpath('mean_amplitudes.txt')

## Data analysis parameters
# WHAT SHOULD BE ANALYZED?
electrodes = ["R2-R7"]                                      # restrict with pick_channels() or drop_channels() if necessary
                                                            # Todo: multiple electrodes will currently result in a channel average of those electrodes
channels_of_interest = ["Fz", "Pz", "Cz", "R2-R7"]          # channels of interest - used to restrict artifact rejection on a subset of electrodes
                                                            # or to drop all other channels for processing speedup
                                                            # WILL NOT BE DONE if correct_artifacts_asr=True

p_epoch = ParamEpochs(time_epoch=(-.100, 0.8),              # Time window to plot [s]
                      time_baseline=(-.100, 0))             # Baseline correction is performed automatically [s]
p3window = [0.3, 0.5]                                       # Extract mean values from here [s]
rej_threshold = 100e-6                                      # represents Maximum PEAK-TO-PEAK amplitude! [µV]
do_reject = True

# SPECIFY ANALYSIS PARAMETERS (Overwriting params.py defaults)
p_preproc = ParamPreprocessing(apply_infinite_reference=False, bandpass=(.1, 30), apply_notch=True, apply_ASR=False)   # was (.1,40)

p_artifact = ParamArtifacts(reject_channels_full_of_artifacts=False,
                            reject_artifactual_epochs=do_reject,  # Todo: classification will not work with non-divisible number of epochs
                            correct_artifacts_asr=False,            # DO NOT USE yet, seems to be implemented twice: p_artifact.correct_artifacts_asr AND p_preproc.apply_ASR
                            artifact_threshold=rej_threshold)  # µV

p_channels = ParamChannels(select_subset=channels_of_interest)


## General
run_offline_classification = False   # Passed to run run_analysis(). Todo: wrap into parameter object thingy if too many parameters pile up
do_GA_analysis = True
p_plots = DisplayPlots(channel_average=True, epochs=False, fixed_display_range=[-2, 6])
p_plots = DisplayPlots(channel_average=True, epochs=False)
Skip_Plot_Visualization = False

# Figures
if Skip_Plot_Visualization:
    import matplotlib
    matplotlib.use('Agg')
#p_int = ParamInterface(export_figures=True)
                # False: Execution halts until plots are closed manually
                # True:  Plots safed, but not shown
                # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib






### Running code #######################################################################################################

evoked_per_subject = []    #  collect averaged epochs ("evoked") per subject

# Loop expects a folder for each session
for name_folder in list_folders:
    logger.info(f"Processing folder {name_folder}")
    path_folder = data_folder.joinpath(name_folder)
    p_data = ParamData(data_dir=path_folder)

    # START the analysis.
    evokeds_out = run_analysis(param_data=p_data,
                               param_preproc=p_preproc,
                               param_artifacts=p_artifact,
                               param_epochs=p_epoch,
                               display_plots=p_plots,
                               param_channels=p_channels,
                               #speller_info=speller_info,
                               current_folder=name_folder,
                               electrodes=electrodes,
                               classify=run_offline_classification)

    evoked_per_subject.append(evokeds_out)          # with ([epochs_out, p_data]):
                                                    # epochs_per_subject[0][1] FOLDER NAME
                                                    # epochs_per_subject[0][0] AVG Epoch


# show a GA plot
# Group Level could also work by collecting ALL files across folders for GA and run the analysis ONCE on ALL files
# CURRENTLY,  implementation has p300analysis return subject averages in a list
# ERROR BAR: CI between entire sessions / VP
if do_GA_analysis:
    from p3k.plots import plot_grand_average_erp
    for electrode in electrodes:
                                      # True: show CI
        fig_GA = plot_grand_average_erp(True, evoked_per_subject, electrode, title="Grand Average") # Todo: bug check plot_grand_average_erp: verify it is really iterating over different evokeds??
        fig_GA[0].savefig(os.path.join('results', '_GA'), dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')

    #from p3k.plots import plot_GA_channel_average
    #plot_GA_channel_average(evoked_per_subject)  # draft - it runs, but shows same signal for all positions? o_O


# Create data frame to store extracted values
p300_table = pd.DataFrame({'VP': pd.Series([], dtype='str'),
                           'Fz': pd.Series([], dtype='float'), 'Cz': pd.Series([], dtype='float'), 'Pz': pd.Series([], dtype='float') })

# retrieve average amplitude per subject average evoked in a time window
for myEvoked in evoked_per_subject:
    current_evoked_array = myEvoked[0]  # Target only # [1] = nonTarget
    Fz=current_evoked_array.get_data(picks="Fz", units='uV', tmin=p3window[0], tmax=p3window[1]).mean()     # calculate mean amplitude from time window, convert to µV
    Cz=current_evoked_array.get_data(picks="Cz", units='uV', tmin=p3window[0], tmax=p3window[1]).mean()
    Pz=current_evoked_array.get_data(picks="Pz", units='uV', tmin=p3window[0], tmax=p3window[1]).mean()
    # Todo: refactor this noob code
    # write line to score table
    line = dict(zip(p300_table.columns, [squeeze(list_folders.pop(0)), Fz, Cz, Pz]))
    p300_table = p300_table.append(line, ignore_index=True)


# save to file
with open(file=extracted_out_file, mode='w') as fi:
    tmp = str(data_folder) + '\nExtraction window: ' + str(p3window) + '\nRejection Threshold (performed?): ' + str(rej_threshold*1E6) + ' (' + str(do_reject) + ')\n\n'
    print(tmp)
    fi.write(tmp)
    fi.write(p300_table.__str__())
print('Values extracted to ' + str(extracted_out_file))




### END



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


''' unused
if False:
    import glob
    data_files = glob.glob("C:/Users/mae08yc/Desktop/P3k Pipeline/Data/VP01_Calibration001" + '/*.dat')
    data_files
    
'''
#p_data = ParamData(data_dir=r'C:\Users\mae08yc\Desktop\P3k Pipeline\Data\tacCalib051')

# !!! Tested only with MNE 1.2.2

from p3k.P300Analysis import run_analysis
from pathlib import Path
from p3k.params import ParamChannels, ParamData, SpellerInfo, \
    ParamPreprocessing, ParamArtifacts, ParamEpochs, ParamInterface, DisplayPlots
import logging
import pandas as pd                          # for data frame
from glob import glob                        # for folder and file listings
from numpy import squeeze

#logging.basicConfig(level=logging.DEBUG)
#logging.debug('debug logging activated')
logger = logging.getLogger(__name__)

# channel names, if not otherwise readable from the data
#p_channels = ParamChannels(cname=['Fz','FC1','FC2','C1','Cz','C2','P3','Pz','P4','Oz'])
# P300 speller description, if not otherwise readable from the data
#speller_info = SpellerInfo(nb_stimulus_rows=7, nb_stimulus_cols=7, nb_seq=10)


# Data locations
data_folder = Path(r"C:\Users\mae08yc\Desktop\P3k Pipeline\Data") #p_data = ParamData(data_dir=r'C:\Users\mae08yc\Desktop\P3k Pipeline\Data\tacCalib051')
data_folder = Path(r"G:\Matthias_Data\STUDIES\Tactile\DATA\Calibrations")

# SUBJECT LEVEL ##
# Specify folders manually:
#list_folders = ["VP04_Calibration001", "VP04_Calibration005"]
#
# Pattern match approach:
match = "VP04_Calibration001"
match = "VP1*005"
list_folders = glob(str(data_folder) + "\\" + match, recursive=False)
list_folders = [item.split("\\")[-1] for item in list_folders]                  # take folder name only -> for entire list: split, take last (-1) element

#list_folders = ["VP04_Calibration001", "VP04_Calibration005"]


# Group Level # this could be used to collect ALL files across folders for GA and run the analysis ONCE on ALL files
# CURRENT implementation has p300analysis return subject averages (from list_folders) in an array or list
if False:
    import glob
    data_files = glob.glob("C:/Users/mae08yc/Desktop/P3k Pipeline/Data/VP01_Calibration001" + '/*.dat')
    data_files


electrodes = ["Cz"]
#electrodes = ["Fz", "Cz", "Pz"]


Skip_Plot_Visualization = True
# Figures
if Skip_Plot_Visualization:
    import matplotlib
    matplotlib.use('Agg')

do_GA_analysis = True



# Epoch time window
p_epoch = ParamEpochs(time_epoch=(-.100, 0.8),
                      time_baseline=(-.100, 0))

# Extract mean values from here
p3window = [0.3, 0.5]

# Deviations from default parameters:
p_preproc = ParamPreprocessing(apply_infinite_reference=False, bandpass=(.1, 40), apply_ASR=False)

p_artifact = ParamArtifacts(reject_channels_full_of_artifacts=False,
                            reject_artifactual_epochs=True,  # Todo: classification will not work with non-divisible number of epochs
                            correct_artifacts_asr=False, # DO NOT USE, evil Loic somehow implemented it twice: p_artifact.correct_artifacts_asr AND p_preproc.apply_ASR
                            artifact_threshold=150e-6)  # µV

p_plots = DisplayPlots(channel_average=True, epochs=False)

run_offline_classification = False #  Todo: wrap into parameter object thingy


# Flavour








#p_int = ParamInterface(export_figures=True)


                # False: Execution halts until plots are closed manually
                # True:  Plots safed, but not shown
                # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib





evoked_per_subject = []
# Loop through all folders # Todo: Return and store
for name_folder in list_folders:
    logger.info(f"Processing folder {name_folder}")
    path_folder = data_folder.joinpath(name_folder)
    p_data = ParamData(data_dir=path_folder)


    # AYY LEZ GOOOO!
    evokeds_out = run_analysis(param_data=p_data,
                 param_preproc=p_preproc,
                 param_artifacts=p_artifact,
                 param_epochs=p_epoch,
                 display_plots=p_plots, #, param_channels=p_channels)  # , speller_info=speller_info)
                 current_folder=name_folder,
                 electrodes=electrodes,
                 classify=run_offline_classification)
    evoked_per_subject.append(evokeds_out)           # with ([epochs_out, p_data]):
                                                    # epochs_per_subject[0][1] FOLDER NAME
                                                    # epochs_per_subject[0][0] AVG Epoch



# show a GA plot
if do_GA_analysis:
    from p3k.plots import plot_grand_average_erp
    plot_grand_average_erp(evoked_per_subject, electrodes) # Todo: bug check plot_grand_average_erp: verify it is really iterating over different evokeds??

    #from p3k.plots import plot_GA_channel_average
    #plot_GA_channel_average(evoked_per_subject)  # draft - it runs, but shows same signal for all positions? o_O


# Create data frame to store extracted values
p300_table = pd.DataFrame({'VP': pd.Series([], dtype='str'),
                           'Fz': pd.Series([], dtype='float'), 'Cz': pd.Series([], dtype='float'), 'Pz': pd.Series([], dtype='float') })

# retrieve average amplitude per subject average evoked in a time window
for myEvoked in evoked_per_subject:
    current_evoked_array = myEvoked[0]  # Target only #  evoked_per_subject[i][0]
    #Fz.append((current_evoked_array.get_data(picks="Fz", units='uV', tmin=p3window[0], tmax=p3window[1])).mean())     # calculate mean amplitude from time window, convert to µV
    Fz=current_evoked_array.get_data(picks="Fz", units='uV', tmin=p3window[0], tmax=p3window[1]).mean()     # calculate mean amplitude from time window, convert to µV
    Cz=current_evoked_array.get_data(picks="Cz", units='uV', tmin=p3window[0], tmax=p3window[1]).mean()
    Pz=current_evoked_array.get_data(picks="Pz", units='uV', tmin=p3window[0], tmax=p3window[1]).mean()
    # Todo: refactor this noob code
    # write line to score table
    line = dict(zip(p300_table.columns, [squeeze(list_folders.pop(0)), Fz, Cz, Pz]))
    p300_table = p300_table.append(line, ignore_index=True)


# save to file
out_name = ('C:/Users/mae08yc/Desktop/P3k Pipeline/Data/_score_table.txt')
with open(file=out_name, mode='w') as fi:
    fi.write(p300_table.__str__())

1+1

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
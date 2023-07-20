import os
from pathlib import Path
from typing import Union, List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import mne
import logging
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

#matplotlib.use('qt5agg')
'''
DEVELOP = False  # todo: make sure it's helpful,
# Set to true if python loads the p3k pip package but want to work on source files directly
# note that it's maybe obsolete
if DEVELOP:
    # this loads p3k into path to prevent python to lookup in site-packages first

    # Get the absolute path of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the p2k folder in your development environment
    p3k_path = Path(current_dir).joinpath("p3k")
    # Insert the p2k folder path at the beginning of sys.path
    sys.path.insert(0, p3k_path)
    sys.path.insert(0, p3k_path.joinpath('classification'))
    sys.path.insert(0, p3k_path.joinpath('read'))
    sys.path.insert(0, p3k_path.joinpath('read/bci_format'))
    sys.path.insert(0, p3k_path.joinpath('signal_processing'))
'''

from p3k import channels
from p3k import epoching
from p3k import plots
from p3k.classification import lda_p3oddball, rsquared
from p3k.params import ParamLDA, ParamInterface, ParamData, ParamEpochs, ParamChannels, InternalParameters, \
    ParamArtifacts, DisplayPlots, SpellerInfo, ParamPreprocessing, SampleParams
from p3k.read import read_eeg
from p3k.signal_processing import artifact_rejection, rereference

from meegkit.asr import ASR  # https://github.com/nbara/python-meegkit
# todo implement artifact subspace reconstruction
logger = logging.getLogger(__name__)


def _make_output_folder(filename_s: Union[str, List[str]], fig_folder: str) -> str:
    if isinstance(filename_s, list):
        output_name = Path(filename_s[0]).stem
    else:
        output_name = Path(filename_s).stem
        if len(filename_s) > 1:
            output_name = output_name + f'_{len(filename_s)}_files'
    print('Figures will have the name: {}'.format(output_name))

    # creating the ./out folder
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    # creating the data related output folder
    fig_folder = os.path.join(fig_folder, output_name)
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)
    print('Created output directory'.format(fig_folder))

    return output_name

# Artifact Subspace Reconstruction (ASR) must be calibrated before it is applied
def fit_asr(data: np.ndarray, fs: int, window_s: float = .5,
            data_interval_s: float = None,
            method: str = 'euclid', estimator: str ='scm'):
    # initialize  the asr model
    asr_model = ASR(sfreq=int(fs), method=method, win_len=window_s, estimator=estimator)

    # if an interval was chosen for training (sec)
    if data_interval_s is not None:
        train_idx = np.arange(data_interval_s[0] * fs, data_interval_s[1] * fs, dtype=int)
    # otherwise select the whole training set
    else:
        train_idx = np.arange(0, data.shape[1])

    train_data = data[:, train_idx]

    # Fit the ASR model with calibration data
    _, sample_mask = asr_model.fit(train_data)
    return asr_model



def run_analysis(param_channels: ParamChannels = None,
                 param_preproc: ParamPreprocessing = None,
                 param_artifacts: ParamArtifacts = None,
                 display_plots: DisplayPlots = None,
                 speller_info: SpellerInfo = None,
                 param_lda: ParamLDA = None,
                 param_interface: ParamInterface = None,
                 param_data: ParamData = None,
                 param_epochs: ParamEpochs = None,
                 internal_params: InternalParameters = None,
                 current_folder: str = None,
                 electrodes: list = None,  # Todo: @matty, check 'param_channels.select_subset'
                 classify: bool = True):  # todo: @matty: when you try to spice up the analysis you may want to keep the vanilla stuff working :)

    if param_channels is None:
        param_channels = ParamChannels()
    if param_preproc is None:
        param_preproc = ParamPreprocessing()
    if internal_params is None:
        internal_params = InternalParameters()
    if param_artifacts is None:
        param_artifacts = ParamArtifacts()
    if display_plots is None:
        display_plots = DisplayPlots()
    if speller_info is None:
        speller_info = SpellerInfo()
    if param_lda is None:
        param_lda = ParamLDA()
    if param_interface is None:
        param_interface = ParamInterface()
    if param_epochs is None:
        param_epochs = ParamEpochs()


    if electrodes is not None:
        raise EidelError('picks in mne refer to the index of the channel, not its name. You should instead use names in in param_channels.cname :)')

    # Checking whether a data folder was provided
    if param_data is None \
            or param_data.data_files is None:
        default_p = SampleParams()

        logger.warning(f"No data folder was specified, "
              f"using the sample file with following parameters {default_p}")
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles | QFileDialog.Directory)
        file_dialog.setNameFilter('Data Files (*.dat *.gdf);;All Files (*)')
        file_dialog.setDirectory('./data/')
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_paths = file_dialog.selectedFiles()
            print(file_paths)
            sel_folder = None
            sel_files = []
            for fp in file_paths:
                pfp = Path(fp)
                if pfp.exists():
                    if pfp.is_dir():
                        sel_folder = pfp
                        break
                    sel_files.append(pfp)

                else:
                    raise FileNotFoundError(pfp)
            # use these parameters
            param_data = ParamData(data_files=file_paths)
        else:
            default_p = ParamData()
            message = f"Using all files in: {default_p.data_dir}\n" \
                      f"Speller info: {default_p.speller_info}\n" \
                      f"channels: {default_p.channels}"
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Icon.Warning)
            message_box.setWindowTitle("Warning")
            message_box.setText(message)
            message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            message_box.exec_()

            # apply those parameters
            param_data = ParamData(data_dir=default_p.data_dir,
                                   extension_filter='dat',
                                   #extension_filter='gdf'
                                   )

            # these parameters should be loaded fronm data files or user defined
            # default_p.speller_info,
            # cname=default_p.channels)

    # Do not import ASR if not used
    if param_preproc.apply_ASR:
        from p3k.signal_processing import ASR

    ## Read the EEG from files
    raw, acquisition_software, speller_info = \
        read_eeg.load_eeg_from_files(eeg_files=param_data.data_files,
                                     begin_stimuli_code=internal_params.STIMULUS_CODE_BEGIN,
                                     speller_info=speller_info)

    assert raw is not None, "Failed to load the files"

    if param_artifacts.correct_artifacts_asr: # Todo: Visualize signal before and after ASR
        # fit the ASR model
        asr_trained = fit_asr(data=raw.get_data(), fs=int(raw.info["sfreq"]))

        # reconstruct data using ASR (reducing artifacts)
        data_reconstructed = asr_trained.transform(raw.get_data())

        # change the data
        raw._data = data_reconstructed

    if acquisition_software == "openvibe":
        assert speller_info.nb_seq is not None, f'If using openvibe data, please define the ' \
                                                f'speller_info = SpellerInfo(<params>) before running '

    # update the detected bci software that generated data
    param_data.acquisition_software = acquisition_software

    # Ensure we have an output folder to save pictures
    if param_interface.export_figures:
        output_name = _make_output_folder(filename_s=raw._filenames,
                                          fig_folder=param_interface.export_figures_path)

    # Channel definition
    raw = channels.define_channels(raw=raw,
                                   channel_names=param_channels.cname,
                                   montage=None)

    if display_plots.montage_plots:
        raw.plot_sensors(show_names=True)
        fig = raw.plot_sensors('3d')

    # Display cross correlation plots
    if display_plots.cross_correlation:
        m = np.corrcoef(raw._data)
        fig = plt.figure()
        hm = sns.heatmap(m, linewidths=0, cmap="YlGnBu").set(title='Cross correlation')

    # display signal before any preprocessing
    if display_plots.raw:
        ep_plot = plots.plot_seconds(raw=raw, seconds=4)



# PREPROCESSING ########################################################################################################

    # Data resampling
    if param_preproc.apply_resample:
        raw.resample(param_preproc.resample_freq)

    # Variance based channel rejection on all the recording
    list_art_ch = artifact_rejection.detect_artifactual_channels(raw=raw, notch_hz=50)
    if param_artifacts.reject_channels_full_of_artifacts:
        channels.flag_channels_as_bad(raw=raw, bad_channels=list_art_ch)

    # Re-referencing to infinite reference
    if param_preproc.apply_infinite_reference:
        raw, _, _, _, _ = rereference.apply_infinity_reference(raw=raw,
                                                               display_plot=display_plots.infinite_reference)

    # Bandpass filtering
    raw = raw.filter(param_preproc.bandpass[0], param_preproc.bandpass[1], fir_window='hann', method='iir')
    #raw = raw.notch_filter(param_preproc.notch)  # removes 50Hz noise
    if display_plots.bandpassed:
        plots.plot_seconds(raw=raw, seconds=10)

    # Detect and reject artifacts in channels
    if param_artifacts.reject_channels_full_of_artifacts:
        rej_ch, fig_art = artifact_rejection.detec_rej_channel(raw=raw,
                                                               threshold_eeg=param_artifacts.artifact_threshold,
                                                               reject_ratio=param_artifacts.ratio_tolerated_artifacts,
                                                               show_plot=display_plots.channel_artifacts)
        if rej_ch is not None:
            channels.flag_channels_as_bad(rej_ch)

    # Apply artifact subspace reconstruction
    if param_preproc.apply_ASR:
        # !pip install meegkit pymanopt
        asr_model = ASR.train_asr(raw)

        raw = ASR.apply_asr(raw=raw,
                            asr_model=asr_model,
                            display_plot=display_plots.asr)

    # Parse and convert the annotations to generate stimuli metadata
    new_annotations, target_map = epoching.parse_annotations(raw.annotations,
                                                             speller_info=speller_info,
                                                             acquisition_software=acquisition_software,
                                                             internal_params=internal_params)
    raw.set_annotations(new_annotations)

    # Convert annotations to events
    all_events, event_id = mne.events_from_annotations(raw, event_id=target_map)
    print("Found {} events".format(len(all_events[:])))

    ### Prepare metadata to annotate events
    # When making target and non-target epochs, we need to conserve stimulus related information.
    # (trial number, stimulus number, column and row information)
    # Metadata is a pandas dataframe with as many rows as there are events, and describes events signal on its columns
    # Display montage
    df_meta = epoching.metadata_from_events(events=all_events,
                                            speller_info=speller_info,
                                            stimulus_code_begin=internal_params.STIMULUS_CODE_BEGIN)

    ### Make epochs
    # Note that the epochs are created based on events. Epochs info is stored in metadata.

    # We only select targets and non targets, those should match exactly with stimuli annotations made in metadata
    events = mne.pick_events(all_events, [0, 1])

    # make epochs AND baseline correct (https://mne.tools/stable/generated/mne.Epochs.html)
    epochs = mne.Epochs(raw, events, baseline=param_epochs.time_baseline,
                        event_id=internal_params.EVENT_IDS,
                        tmin=param_epochs.time_epoch[0], tmax=param_epochs.time_epoch[1],
                        event_repeated='drop', picks=['eeg', 'csd'],
                        preload=True,
                        metadata=df_meta)

    if display_plots.epochs:
        fig = epochs[0:5].plot(title='displaying 5 first epochs')

    ### Epoch rejection
    # Channels should be filtered out before epochs because any faulty channel would cause every epoch to be discarded
    if param_artifacts.reject_artifactual_epochs:
        reject_criteria = dict(eeg=param_artifacts.artifact_threshold)  # 100 µV  #eog=200e-6)
        epochs = epochs.drop_bad(reject=reject_criteria) # Todo
        #if display_plots.reject_epochs:
         #   epochs.plot_drop_log()

    ## Apply current source density
    if param_preproc.apply_CSD:
        print("Applying CSD")
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
        epochs = epochs_csd
        if display_plots.csd:
            fig = epochs_csd[0:5].plot(title='Current_source_density on 5 first epochs')

    ### Epochs aggregation and display
    # Classwise butterfly with topomap
    if display_plots.butterfly_topomap:
        plots.plot_butterfly_topomap(epochs=epochs)

    # classwise averages
    if display_plots.channel_average:
              #fig = plots.plot_channel_average(epochs=epochs)   # OVERVIEW of all electrodes - bugfixed, would sometimes find only 1 axis tick and then crash.
        #fig_avg_ERP = plots.plot_average_erp(epochs=epochs, title=current_folder)#, picks=electrodes)[0]               # Average w/o CI (faster)
        list_fig_ERP = plots.plot_CI_erp(epochs=epochs, title=current_folder,
                                    ch_subset=param_channels.select_subset, average_channels=False,
                                         display_range=display_plots.fixed_display_range) # Average with CI
        #fig_ERP.set_size_inches(4, 3)     # error because code halts during figure display, no ref after figure closed manually. Todo: Put into function before figure shown
        if param_interface.export_figures:
            out_folder = Path(param_interface.export_figures_path).joinpath(f'{output_name}')
            if isinstance(list_fig_ERP, list):
                for dict_fig in list_fig_ERP:
                    out_filepath = out_folder.joinpath(f'{output_name}_{dict_fig["lbl"]}')
                    dict_fig['ax'].savefig(out_filepath, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')
            else:
                out_filepath = out_folder.joinpath(f'{output_name}')
                list_fig_ERP.savefig(out_filepath, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')

    # single trial heatmaps
    if display_plots.erp_heatmap:
        plots.plot_erp_heatmaps(epochs=epochs)

    # channelwise heatmaps
    if display_plots.erp_heatmap_channelwise:
        plots.plot_erp_heatmaps_channelwise(epochs=epochs,
                                            csd_applied=param_preproc.apply_CSD)

    ### Rsquared
    if display_plots.signed_r_square:
        rsq, fig_rsq = rsquared.signed_r_square(epochs=epochs,
                                                time_epoch=param_epochs.time_epoch,
                                                display_rsq_plot=True)

        if param_interface.export_figures:
            out_name = os.path.join(param_interface.export_figures_path, output_name,
                                    output_name + '_rsquared')
            fig_rsq.savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')

    # Classify, or don't
    if not classify:
        avg_evoked = epoching.get_avg_target_nt(epochs) # returns evoked tuple: T/NT
        return avg_evoked                               # todo: restrict to picks







    ### Classification LDA                                          # currently not working with dropped epochs
    # resample for faster lda

    if param_lda.resample_LDA is not None:
        new_fs = param_lda.resample_LDA  #
        epochs = epochs.copy().resample(new_fs)
        print('resampling to {}Hz'.format(new_fs))

    ## Single epoch classification
    cum_score_table = lda_p3oddball.run_p300_LDA_analysis(epochs=epochs,
                                                          nb_k_fold=param_lda.nb_cross_fold,
                                                          speller_info=speller_info)

    ## Prediction results in table
    # **fold**: k-fold\
    # **fold_trial**: index of trial contained in the fold\
    # **n_seq**: number of sequences selecting epoch for predictions\
    # **score**: LDA score (target vs non-target detection). Classes are unbalanced so the score is misleading\
    # **AUC**: LDA Area Under the Curve. Estimation of the performance of the classifier\
    # **row/col_pred/true**: row and columns target and predicted\
    # **correct**: the predicted row **AND** column is correctly predicted
    if display_plots.offline_accuracy:
        fig_score = lda_p3oddball.plot_cum_score_table(table=cum_score_table,
                                                       nb_cross_fold=param_lda.nb_cross_fold)
        if param_interface.export_figures:
            out_name = os.path.join(param_interface.export_figures_path, output_name,
                                    output_name + '_accuracy')
            fig_score.savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')

    print(f"Number of ERP targets={epochs['Target']._data.shape[0]},"
          f" non-targets={epochs['NonTarget']._data.shape[0]}")

    # save the table # Hi Loic! You rock!!1
    if display_plots.score_table:
        out_name = os.path.join(param_interface.export_figures_path, output_name,
                                output_name + '_score_table.txt')

        print(cum_score_table)
        with open(file=out_name, mode='w') as fi:
            # Disable pandas limitations
            bak_max_rows = pd.options.display.max_rows
            bak_max_columns = pd.options.display.max_columns
            bak_width = pd.options.display.width
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            # save to file
            fi.write(cum_score_table.__str__())
            # Restore Pandas limitations
            pd.set_option('display.max_rows', bak_max_rows)
            pd.set_option('display.max_columns', bak_max_columns)
            pd.set_option('display.width', bak_width)
            print(f"saved to file {out_name}")


class EidelError(Exception):
    pass


if __name__ == "__main__":
    # test using sample data
    #run_analysis()
    # use qt5 the file popup dialog
    app = QApplication(sys.argv)

    example = 'bci2000_nochannel'
    example = 'openvibe'
    example = 'bci2000_onerow'
    example = None

    if True:
        # data_files can be a str or a list of str of files and folders
        p_channels = ParamChannels()

        # info about the speller, autodetected for BCI2000 but not for OpenVibe
        p_speller_info = SpellerInfo(nb_seq=None,
                                     nb_stimulus_rows=None,
                                     nb_stimulus_cols=None)

        if example == 'bci2000_nochannel':
            # load files in a folder according filename and extension filters
            p_data = ParamData(data_files=r'./data_sample',
                               filename_filter=".*",
                               extension_filter='dat',
                               )
            # channels were not well defined in this data file, so we must specify them
            p_channels = ParamChannels(['Fz', 'Cz',
                                        'P3', 'Pz', 'P4',
                                        'O1', 'Oz', 'O2'])
        elif example == 'openvibe':
            # load a single file
            p_data = ParamData(data_files=r'./data_sample/loic_gammabox.gdf',
                               #filename_filter=".*",
                               #extension_filter='gdf',
                               )
            # in openvibe there is no information about the channels
            p_channels = ParamChannels(cname=SampleParams().channels)
            # nor about the nb of sequences, rows and columns, use the sample
            p_speller_info = SampleParams().speller_info

        elif example == "bci2000_onerow":
            p_data = ParamData(data_files='SH-AuditoryOddball001')

        else:
            logger.info("No data files selected")
            p_data = ParamData()
            pass



        p_epochs = ParamEpochs(time_epoch=(-.2, 1),
                               time_baseline=(-.2, 0))

        p_preproc = ParamPreprocessing(apply_infinite_reference=False,
                                       apply_CSD=False,
                                       apply_ASR=False)

        p_lda = ParamLDA(nb_cross_fold=5)

        d_plots = DisplayPlots(erp_heatmap=False,
                               butterfly_topomap=False,
                               )

        run_analysis(param_data=p_data,
                     param_channels=p_channels,
                     speller_info=p_speller_info,
                     param_preproc=p_preproc,
                     param_epochs=p_epochs,
                     param_lda=p_lda,
                     display_plots=d_plots)

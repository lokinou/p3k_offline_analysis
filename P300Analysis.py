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
logging.basicConfig(level=logging.DEBUG)
logging.debug('debug logging activated')

from p3k import channels
from p3k import epoching
from p3k import plots
from p3k.classification import lda_p3oddball, rsquared
from p3k.params import ParamLDA, ParamInterface, ParamData, ParamEpochs, ParamChannels, InternalParameters, \
    ParamArtifacts, DisplayPlots, SpellerInfo, ParamPreprocessing, SampleParams
from p3k.read import read_eeg
from p3k.signal_processing import artifact_rejection, rereference

from meegkit.asr import ASR

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
                 internal_params: InternalParameters = None):
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

    # Checking whether a data folder was provided
    if param_data is None:
        default_p = SampleParams()
        print(f"!!! WARNING !!! No data folder was specified, "
              f"using the sample file with following parameters {default_p}")
        # apply those parameters
        param_data = ParamData()
        param_data.data_dir = default_p.data_dir
        speller_info = default_p.speller_info
        param_channels.cname = default_p.channels

    # Do not import ASR if not used
    if param_preproc.apply_ASR:
        from p3k.signal_processing import ASR

    ## Read the EEG from files

    raw, acquisition_software, speller_info = \
        read_eeg.load_eeg_from_folder(data_path=param_data.data_dir,
                                      begin_stimuli_code=internal_params.STIMULUS_CODE_BEGIN,
                                      speller_info=speller_info)

    if param_artifacts.correct_artifacts_asr:
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

    assert raw is not None, "Failed to load the files"

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
        ep_plot = plots.plot_seconds(raw=raw, seconds=10)

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
    raw = raw.notch_filter(param_preproc.notch)  # removes 50Hz noise
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

    # make epochs
    epochs = mne.Epochs(raw, events, baseline=param_epochs.time_baseline,
                        event_id=internal_params.EVENT_IDS,
                        tmin=param_epochs.time_epoch[0], tmax=param_epochs.time_epoch[1],
                        event_repeated='drop', picks=['eeg', 'csd'],
                        preload=True,
                        metadata=df_meta)

    if True or display_plots.epochs:
        fig = epochs[0:5].plot(title='displaying 5 first epochs')

    ### Epoch rejection
    # Channels should be filtered out before epochs because any faulty channel would cause every epoch to be discarded
    if param_artifacts.reject_artifactual_epochs:
        reject_criteria = dict(eeg=param_artifacts.artifact_threshold)  # 100 µV  #eog=200e-6)
        _ = epochs.drop_bad(reject=reject_criteria) # Todo
        if display_plots.reject_epochs:
            epochs.plot_drop_log()

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
        fig = plots.plot_channel_average(epochs=epochs, )
        if param_interface.export_figures:
            out_name = os.path.join(param_interface.export_figures_path, output_name,
                                    output_name + '_ERPs')
            fig.savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')

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

    ### Classification LDA
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

    # save the table
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


if __name__ == "__main__":
    # test using sample data
    #run_analysis()

    if True:
        p_data = ParamData(data_dir=r'./SH-tac_oddball001')

        p_epochs = ParamEpochs(time_epoch=(-.2, 1),
                               time_baseline=(-.2, 0))

        p_preproc = ParamPreprocessing(apply_infinite_reference=False,
                                       apply_CSD=False,
                                       apply_ASR=False)

        p_lda = ParamLDA(nb_cross_fold=5)

        d_plots = DisplayPlots(erp_heatmap=False,
                               butterfly_topomap=False)

        run_analysis(param_data=p_data,
                     param_preproc=p_preproc,
                     param_epochs=p_epochs,
                     param_lda=p_lda,
                     display_plots=d_plots)

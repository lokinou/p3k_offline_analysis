import os
from pathlib import Path
from typing import Union, List, Tuple

from p3k.params import ParamLDA, ParamInterface, ParamData, ParamEpochs, ParamChannels, InternalParameters, \
    ParamArtifacts, DisplayPlots, SpellerInfo, ParamPreprocessing

from p3k.read import read_eeg
from p3k import channels
from p3k.signal_processing import artifact_rejection, rereference, ASR
from p3k import plots
from p3k import epoching


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
    if param_data is None:
        param_data = ParamData()
    if param_epochs is None:
        param_epochs = ParamEpochs()

    ## Read the EEG from files

    raw, acquisition_software, speller_info = \
        read_eeg.load_eeg_from_folder(data_path=param_data.data_dir,
                                      begin_stimuli_code=internal_params.STIMULUS_CODE_BEGIN,
                                      speller_info=speller_info)

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
    import mne
    # Channel subsetting
    print(f"todo: select a subset of channels select_subset")
    if param_channels.select_subset is not None:
        print(f"selecting {param_channels.select_subset}")
        #intersects = True  # all subset found in the channel list
        #assert intersects, "todo verification"
        raw = raw.pick_channels(ch_names=param_channels.select_subset)
        #print(f"selected {ParamChannels.select_subset}")
        pass

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
        raw, _, _, _, _ = rereference.apply_infinite_reference(raw=raw,
                                                               display_plot=display_plots.infinite_reference)

    # Bandpass filtering
    raw.filter(.5, 40, fir_window='hann', method='iir')
    raw.notch_filter(50)  # removes 50Hz noise
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
        #!pip install meegkit pymanopt
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

if __name__ == "__main__":
    # Define the study parameters
    param_channels = ParamChannels(cname=['Fz', 'FC1', 'FC2', 'C1', 'Cz', 'C2',
                                          'P3', 'Pz', 'P4', 'Oz'])
    param_channels = ParamChannels(cname=['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz'])
                                   #select_subset=['Fz', 'Cz', 'P3'])

    # Run the analysis with the parameters
    run_analysis(param_channels=param_channels)

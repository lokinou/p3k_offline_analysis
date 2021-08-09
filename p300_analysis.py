if __name__ == "__main__":


    # In[2]:


    # if packages are missing, uncomment and execute here or in anaconda prompt with p300mne env
    #!pip install "git+https://github.com/nbara/python-meegkit"
    #!pip install statsmodels pyriemann


    # In[3]:


    from matplotlib import pyplot as plt
    import numpy as np
    import os
    import mne
    import seaborn as sns
    # LDA

    from p3k.read import read
    from p3k import offline_analysis
    from p3k.offline_analysis import SpellerInfo, DisplayPlots
    from p3k.classification import lda_p3oddball


    # ## Parameters

    # I don't know any python package to read .ov files, you must convert them. Check my [ov to gdf tutorial](https://github.com/lokinou/openvibe_to_gdf_tutorial)\
    #
    # <span style="color:red">**Before you execute the script**</span>, make sure do to double check the follwing:
    # - electrodes names (`cname`)
    # - Baseline and epoch durations (`time_baseline, time_epoch`)
    # - Cross fold splits (`nb_k_splits`) must be a multiple of the number of trials
    # - For OpenVibe data, Manully define `SpellerInfo` (for row, col and nb sequences)

    # ## Define global parameters here

    # In[4]:


    # Directory containing the gdf files
    data_dir = r"./data_sample"  # folder is scanned for .gdf or .dat files

    # Define the electrodes here (for the provided sample file)
    cname = None
    cname = ['Fz', 'FC1', 'FC2', 'C1', 'Cz', 'C2', 'P3', 'Pz', 'P4', 'Oz']  #bci2000 sample
    #cname = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']  # openvibe sample
    #cname = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz', 'bad1', 'bad2']  # openvibe sample
    # ERP analysis parameters (values in sec)
    display_channel_erp = ["Cz"]
    time_epoch = (-.200, .600)  # epoch (tmin, tmax) 200ms prior to stimulus
    time_baseline = (-.200, 0)  # (baseline_tmin, baseline_tmax),  should contain the baseline

    # resample for faster processing
    resample_freq = None
    resample_freq = 256  # Hz


    # In[5]:


    # LDA
    resample_LDA = 32  # Hz, dramatically speeds up LDA training and classification
    nb_cross_fold = 5  # number of trials in files must be a mu

    # Decide which figures to display
    #display_plots = DisplayPlots()  # alternatively using default values
    display_plots = DisplayPlots(raw=False,
                                 infinite_reference=False,
                                 bandpassed=False,
                                 asr=True,
                                 csd=False,
                                 cross_correlation=False,
                                 epochs=True,
                                 reject_epochs=True,
                                 butterfly=False,
                                 butterfly_topomap=True,
                                 channel_average=True,
                                 erp_heatmap=False,
                                 erp_heatmap_channelwise= False,
                                 signed_r_square=True,
                                 best_channel_erp=True
    )
    skip_slow_ERP_plot = False  # skip the channelwise ERP plot
    export_figures = True

    reject_channels_full_of_artifacts = False
    reject_artifactual_epochs = False

    artifact_threshold = 100e-6
    ratio_tolerated_artifacts = 0.3  # if 30% of artifacts in 200ms windows, then the channel is rejected

    # OpenVibe parameters
    if True:  # using OpenVibe
        #Setup the speller information for OpenVibe
        speller_info = SpellerInfo(nb_stimulus_rows=7,
                                   nb_stimulus_cols=7,
                                   nb_seq=10)


    # #### Constants

    # In[6]:


    STIMULUS_CODE_BEGIN = 100
    EVENT_IDS = dict(NonTarget=0, Target=1)


    # #### advanced parameters

    # In[7]:


    # Signal Preprocessing
    apply_infinite_reference = False  # re-referencing
    apply_ASR = False  # use Artifact Subspace Reconstruction (artifact removal)
    apply_CSD = False  # use Current Source Density (spatial filter)

    fig_folder = './out'  # output for figures


    # In[8]:


    # For internal processing, stimuli begin at 100 to discriminate from MNE usually using stimuli 1 and 0 as target and non-target
    apply_resample = resample_freq is not None
    stimulus_padding = 100
    acquisition_software = None  # bci2000 or openvibe or None for autodetection


    # ## Load the data files

    # In[9]:


    #fn = ["./data_sample/bci2000\Heide_einsteinBP_calibration4S001R01.dat"]
    # Load data from the folder
    raw, acquisition_software, speller_info = read.load_eeg_from_folder(data_path=data_dir,
                                                                        speller_info=speller_info)


    # In[10]:


    speller_info.__repr__()


    # Create a name for figures output

    # In[11]:


    output_name = offline_analysis.make_output_folder(filename_s=raw._filenames,
                                                      fig_folder=fig_folder)


    # #### Detect units for EEG
    # force the signal to be expressed in Volts (default for MNE)

    # In[12]:


    # If the variance of the data is >1, it means the data is expressed in microvolts
    # Since MNE uses Volt as a default value, we rescale microvolts to volt
    raw = offline_analysis.rescale_microvolts_to_volt(raw)


    # ## Resample

    # In[13]:


    if apply_resample:
        raw.resample(resample_freq)


    # In[14]:


    montage = None  # you can define a specific montage here, otherwise using 10-05 as default

    raw, montage = offline_analysis.define_channels(raw=raw,
                                                    channel_names=cname,
                                                    montage=montage)
    raw = raw.set_montage(montage, match_case=False)


    # Check whether there are bad channels

    # In[15]:


    list_art_ch = offline_analysis.detect_artifactual_channels(raw=raw, notch_hz=50)


    # In[16]:


    if display_plots.raw:
        ep_plot = offline_analysis.plot_seconds(raw=raw, seconds=10)


    # rereferencing

    # In[17]:


    if apply_infinite_reference:
        raw = offline_analysis.apply_infinite_reference(raw=raw,
                                                        display_plot=display_plots.infinite_reference)


    # ## Bandpass the signal
    # Removes noise and drift from the EEG signal by applying a infinite impulse response (two-pass) filter between .5 and 40Hz

    # In[18]:


    raw.filter(.5, 40, fir_window='hann', method='iir')
    raw.notch_filter(50)  # removes 50Hz noise
    if display_plots.bandpassed:
        offline_analysis.plot_seconds(raw=raw, seconds=10)


    # ## Excluding of channels full of artifacts (muscular or disconnecting)
    #
    #

    # In[19]:


    reject_channels_full_of_artifacts = True

    if reject_channels_full_of_artifacts:
        rej_ch = offline_analysis.detec_rej_channel(raw=raw,
                                                    threshold_eeg=artifact_threshold,
                                                    reject_ratio=ratio_tolerated_artifacts,
                                                    show_plot=True)
        if rej_ch is not None:
            offline_analysis.flag_channels_as_bad(rej_ch)


    # ## Artifact Subspace Reconstruction fitting and reconstruction

    # In[20]:


    if apply_ASR:
        #!pip install meegkit pymanopt
        asr_model = offline_analysis.train_asr(raw)

        raw = offline_analysis.apply_asr(raw=raw,
                                         asr_model=asr_model,
                                         display_plot=display_plots.asr)


    # ### Convert text annotations (i.e. unprocessed events) into events

    # **Small but major hack to realign events due to conversion**
    #
    #

    # In[21]:


    # Parse annotations with the follwing mapping
    # non-target=0, target=1, new_trial=10 and stimulus_1=101
    new_annotations, target_map = offline_analysis.parse_annotations(raw.annotations,
                                                                     speller_info=speller_info,
                                                                     acquisition_software=acquisition_software,
                                                                     stimulus_code_begin=STIMULUS_CODE_BEGIN)
    raw.set_annotations(new_annotations)


    # In[22]:


    target_map


    # Then we can convert annotations into events

    # In[23]:


    all_events, event_id = mne.events_from_annotations(raw, event_id=target_map)
    print("Found {} events".format(len(all_events[:])))
    event_id


    # In[24]:


    raw.info["ch_names"]


    # ### Pick the channels

    # In[25]:


    # pick all channels
    picks = mne.pick_channels(raw.info["ch_names"], include=[])
    picks
    raw.plot_sensors(show_names=True)
    fig = raw.plot_sensors('3d')


    # ## Epoching from events

    # ### Prepare metadata to annotate events
    # When making target and non-target epochs, we need to conserve stimulus related information.
    # (trial number, stimulus number, column and row information)
    #
    # Metadata is a pandas dataframe with as many rows as there are events, and describes events signal on its columns

    # In[26]:


    df_meta = offline_analysis.metadata_from_events(events=all_events,
                                                    speller_info=speller_info,
                                                    stimulus_code_begin=STIMULUS_CODE_BEGIN)


    df_meta


    # In[26]:





    # ### Make epochs
    # Note that the epochs are created based on the events.
    # We only select targets and non targets, those should match exactly with stimuli annotations made in metadata

    # In[27]:


    # since we use metadata we can pick only target and non-target events
    events = mne.pick_events(all_events, [0, 1])



    # epoching function
    epochs = mne.Epochs(raw, events, baseline=time_baseline,
                        event_id=EVENT_IDS,
                        tmin=time_epoch[0], tmax=time_epoch[1],
                        event_repeated='drop', picks=['eeg', 'csd'],
                        preload=True,
                        metadata=df_meta)

    # if there is any delay,
    #epochs.shift_time(-isi, relative=True)
    if display_plots.epochs:
        fig = epochs[0:5].plot(title='displaying 5 first epochs')


    # ### Making a cross correlation plot between the electrodes to see how channels relate

    # In[28]:


    if display_plots.cross_correlation:

        m = np.corrcoef(raw._data)
        fig = plt.figure()
        hm = sns.heatmap(m, linewidths=0, cmap="YlGnBu").set(title='Cross correlation')


    # ### Epoch rejection
    # Please filter out channels before epochs. A problematic channel can discard the whole recording

    # In[29]:



    if reject_artifactual_epochs:
        reject_criteria = dict(eeg=artifact_threshold)  # 100 µV  #eog=200e-6)
        _ = epochs.drop_bad(reject=reject_criteria)
        if display_plots.reject_epochs:
            epochs.plot_drop_log()


    # ## Apply current source density

    # In[30]:


    if apply_CSD:
        print("Applying CSD")
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
        epochs = epochs_csd
        if display_plots.csd:
            fig = epochs_csd[0:5].plot(title='Current_source_density on 5 first epochs')


    # ### Average the epochs of each class

    # In[30]:





    # target and non target signal plots

    # joint plot (of the two former graphs). Plase not that Y scales differ between plots

    # In[31]:


    if display_plots.butterfly_topomap:
        offline_analysis.plot_butterfly_topomap(epochs=epochs)


    # ### Target vs NonTarget Erps per channel

    # In[32]:


    if display_plots.channel_average:
        fig = offline_analysis.plot_channel_average(epochs=epochs)

    if export_figures:
        out_name = os.path.join(fig_folder, output_name + '_ERPs')
        fig.savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')


    # ### Display single epochs

    # In[33]:


    if display_plots.erp_heatmap:
        offline_analysis.plot_erp_heatmaps(epochs=epochs)


    # ### Same plot but channel wise

    # In[34]:


    if display_plots.erp_heatmap_channelwise:
        offline_analysis.plot_erp_heatmaps_channelwise(epochs=epochs, csd_applied=apply_CSD)


    # # Classical LDA training

    # resample the signal, we don't need that much

    # In[35]:


    new_fs = resample_LDA  #
    epochs_resampled = epochs.copy().resample(new_fs)
    print('resampling to {}Hz'.format(new_fs))


    #

    # In[35]:





    # In[36]:


    fig_conf, fig_roc = offline_analysis.run_single_epoch_LDA_analysis(X_data=epochs_resampled._data,
                                                                       y_true_labels=epochs_resampled.events[:, 2],
                                                                       nb_k_fold=nb_cross_fold)

    if export_figures:
        out_name = os.path.join(fig_folder, output_name + '_confidence_matrix')
        fig_conf.savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')
        out_name = os.path.join(fig_folder, output_name + '_ROC')
        fig_roc.savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')


    # ## Process the ROC curve and precision recall
    #

    # ## Signed R-Square plot
    # function adapted from wyrm

    # In[37]:


    rsq = None
    if display_plots.signed_r_square:
        rsq, fig_rsq = offline_analysis.signed_r_square(epochs=epochs,
                                                        time_epoch=time_epoch,
                                                        display_rsq_plot=display_plots.signed_r_square)
        if export_figures:
            out_name = os.path.join(fig_folder, output_name + '_heatmap')
            fig_rsq.savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')


    # ### make a pandas database to properly display electrodes and samples
    #

    # In[38]:


    import importlib

    importlib.reload(offline_analysis)
    importlib.reload(offline_analysis.bci_softwares.openvibe)
    #all_events.shape


    # ### Quickly Display a channel with max rsq

    # In[39]:


    if rsq is not None and display_plots.signed_r_square and display_plots.best_channel_erp:
        if display_channel_erp is None:
            ch_max, _ = np.where(rsq == np.max(rsq))
            display_channel_erp = epochs.info['ch_names'][int(ch_max)]

        #picks = [f'eeg{n}' for n in range(10, 15)]
        #evokeds = dict(NonTarget=list(epochs['NonTarget'].iter_evoked()),
        #               Target=list(epochs['Target'].iter_evoked()))
        axs = offline_analysis.plot_average_erp(epochs=epochs, picks=display_channel_erp)

        if export_figures:
            out_name = os.path.join(fig_folder, output_name + '_best_channel')
            axs[0].savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')


    # # Extract offline analysis, using shrinkage LDA

    # ### Extract correct target pair

    # In[39]:





    # #### Custom functions to extract performance in a cumulative way
    # By cumulative, it uses either the first sequence, then first and second, and so on until all sequences are considered.\
    # This allows for choosing an optimal number of sequences

    # In[ ]:





    # #### Resampling for faster LDA

    # In[40]:


    new_fs = resample_LDA  #
    epochs_resampled = epochs.copy().resample(new_fs)
    print('resampling to {}Hz'.format(new_fs))


    # #### Checking whether the number of folds matches the total number of trials

    # In[41]:


    # Check whether we can proceed to k split


    # #### LDA for P300 speller target prediction

    # In[42]:


    score_table = lda_p3oddball.run_p300_LDA_analysis(epochs=epochs_resampled,
                                                      nb_k_fold= nb_cross_fold,
                                                      speller_info=speller_info)
    import numpy as np


    # #### Prediction results
    # **fold**: k-fold\
    # **fold_trial**: index of trial contained in the fold\
    # **n_seq**: number of sequences selecting epoch for predictions\
    # **score**: LDA score (target vs non-target detection). Classes are unbalanced so the score is misleading\
    # **AUC**: LDA Area Under the Curve. Estimation of the performance of the classifier\
    # **row/col_pred/true**: row and columns target and predicted\
    # **correct**: the predicted row **AND** column is correctly predicted

    # In[ ]:


    score_table[:]


    # #### Make a human readable plot of these results

    # In[ ]:


    df_seq = score_table.groupby(['n_seq']).mean()
    df_seq = df_seq.rename(columns={"correct": "Accuracy", "score": "epoch_score", "AUC": "epoch_AUC"})
    df_seq[['Accuracy', 'epoch_score', 'epoch_AUC']]

    df_seq.plot(y='Accuracy')
    plt.ylim(0, 1.01)
    plt.suptitle('Cross-fold offline accuracy (n={})'.format(nb_k_splits))
    plt.xlabel('Number of sequences')
    plt.ylabel('Accuracy')
    plt.legend(['P300 target prediction (row x col)'])
    # export the figure
    out_name = os.path.join(fig_folder, output_name + '_accuracy')
    plt.savefig(out_name, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')
    plt.show()
    print("Number of ERP targets={}, non-targets={}".format(epochs['Target']._data.shape[0],
                                                            epochs['NonTarget']._data.shape[0]))


    # In[ ]:





    # In[ ]:





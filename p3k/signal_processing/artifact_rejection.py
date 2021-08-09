#Apply a variance based channel rejection if artifacts are present >30% of the time
def detec_rej_channel(raw, threshold_eeg: float,
                      reject_ratio: float,
                      window_sec=.2,
                      overlap_sec=.1,
                      show_plot=False):
    epochs_rej = mne.make_fixed_length_epochs(raw, duration=window_sec, overlap=overlap_sec, preload=True)
    epochs_rej._data.shape
    diff = np.max(epochs_rej._data, axis=2) - np.min(epochs_rej._data, axis=2)

    print(diff.shape)

    rej = (diff >= threshold_eeg).astype(np.float64)
    rel = sns.heatmap(rej)
    rel.set(title='Detected artifacts per electrode and runs (white)')

    # calculate ratio of rejected trials
    ratios = np.sum(rej, axis=0) / rej.shape[0]

    ret = np.argwhere(ratios >= reject_ratio).tolist()
    if len(ret) > 0 and len(ret[0]):
        print('Found {} channels with at least {}% {}s epochs > {} amplitude)'.format(len(ret),
                                                                                      reject_ratio * 100, window_sec,
                                                                                      threshold_eeg))
        return ret[0]
    else:
        return None

def detect_artifactual_channels(raw: mne.io.BaseRaw,
                                notch_hz: int = 50):
    # Using 50Hz power variance
    psd, freqs = mne.time_frequency.psd_welch(raw, verbose=True)
    power_50hz = psd[:, np.where(freqs == notch_hz)]
    ch_art_line_var = mne.preprocessing.bads._find_outliers(power_50hz.squeeze(), threshold=3, max_iter=5, tail=0)
    print(f'{notch_hz}Hz variance rejection: {ch_art_line_var}')

    # using variance
    ch_var = [np.var(raw._data[i, :]) for i in list(range(raw._data.shape[0]))]
    ch_art_var = mne.preprocessing.bads._find_outliers(ch_var, threshold=3, max_iter=5, tail=0)
    print(f'Variance based rejection: {ch_art_var}')

    return list(set(ch_art_line_var + ch_art_var))

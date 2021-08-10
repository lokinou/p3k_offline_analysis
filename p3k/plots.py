import mne
import sklearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from p3k import epoching
from sklearn.metrics import roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay


def plot_seconds(raw: mne.io.BaseRaw, seconds: float, title: str = None):
    fig = mne.make_fixed_length_epochs(raw, duration=seconds)[0].plot()
    if title is not None:
        fig.suptitle(title,  size='xx-large', weight='bold')

    return fig

def plot_butterfly(epochs: mne.Epochs):
    l_target, l_nt = epoching.get_avg_target_nt(epochs=epochs)

    fig, ax = plt.subplots(2, 1)
    ax1 = l_target.plot(spatial_colors=True, axes=ax[0], show=False)
    ax2 = l_nt.plot(spatial_colors=True, axes=ax[1], show=False)
    # Add title
    fig.suptitle("Target(top) - Non-Target(bottom)")
    # Fix font spacing
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return fig, l_target, l_nt



def plot_topomap(epochs: mne.Epochs):
    l_target, l_nt = epoching.get_avg_target_nt(epochs=epochs)
    spec_kw = dict(width_ratios=[1, 1, 1, .15], wspace=0.5,
                   hspace=0.5, height_ratios=[1, 1])
    # hspace=0.5, height_ratios=[1, 2])

    fig, ax = plt.subplots(2, 4, gridspec_kw=spec_kw)
    l_target.plot_topomap(times=[0, 0.18, 0.4], average=0.05, axes=ax[0, :], show=False)
    l_nt.plot_topomap(times=[0, 0.18, 0.4], average=0.05, axes=ax[1, :], show=False)
    fig.suptitle("Target(top) - Non-Target(bottom)")
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return fig

def plot_butterfly_topomap(epochs: mne.Epochs):
    l_target, l_nt = epoching.get_avg_target_nt(epochs=epochs)
    l_target.plot_joint()
    fig_target = plt.gcf().canvas.set_window_title('Target joint plot')
    l_nt.plot_joint()
    figt_nt = plt.gcf().canvas.set_window_title('Non-Target joint plot')
    return fig_target, figt_nt


def plot_average_erp(epochs: mne.Epochs, picks=None):
    l_target, l_nt = epoching.get_avg_target_nt(epochs=epochs)
    evokeds = dict(NonTarget=l_nt,
                   Target=l_target)
    # picks = [f'eeg{n}' for n in range(10, 15)]
    fig_handle = mne.viz.plot_compare_evokeds(evokeds, picks=picks, combine='mean')
    return fig_handle


def plot_channel_average(epochs: mne.Epochs):
    nb_chans = epochs['Target']._data.shape[1]
    splt_width = int(np.floor(
        np.sqrt(1.0 * nb_chans + 2)))  # adding an extra plot with all channels combined at the end and a legend
    splt_height = splt_width if splt_width * splt_width >= nb_chans + 1 else splt_width + 1
    while splt_height * splt_width < nb_chans + 2:
        splt_height += 1
    fig, axes = plt.subplots(splt_height, splt_width, figsize=(10, 8), sharex=True, sharey=True)

    # evokeds = dict(NonTarget=list(epochs['NonTarget'].iter_evoked()),
    #               Target=list(epochs['Target'].iter_evoked()))
    evokeds = dict(NonTarget=epochs['NonTarget'].average(),
                   Target=epochs['Target'].average())
    # picks = [f'eeg{n}' for n in range(10, 15)]

    shape_epochs = epochs['Target']._data.shape
    nb_cells = splt_height * splt_width
    for plot_idx in range(nb_cells):

        # cells containing data
        if plot_idx in range(nb_chans):
            ch_idx = plot_idx
            print('plotting channel {}'.format(ch_idx + 1))
            mne.viz.plot_compare_evokeds(evokeds, picks=[epochs.info['ch_names'][ch_idx]],
                                         legend=False,
                                         axes=axes[plot_idx // splt_width, plot_idx % splt_width], show=False)
            # plt.show(block=False)
            plt.subplots_adjust(hspace=0.5, wspace=.5)
            # plt.pause(.05)

        # filler and legend cells
        elif plot_idx <= nb_cells - 2:
            ax = axes[plot_idx // splt_width, plot_idx % splt_width]
            ax.clear()  # clears the random data I plotted previously
            ax.set_axis_off()  # removes the XY axes

            if plot_idx == nb_cells - 2:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                leg = ax.legend(handles, labels)

    print('plotting averaged channels')
    ax_evoked = mne.viz.plot_compare_evokeds(evokeds, picks=epochs.info['ch_names'], combine='mean',
                                     legend=False,
                                     axes=axes[-1, -1], show=False)

    # retrieve the legend and move it in the previous cell

    plt.subplots_adjust(hspace=1, wspace=.1)
    plt.show()
    return fig

def plot_erp_heatmaps(epochs: mne.Epochs):
    epochs['Target'].plot_image(combine='mean')
    fig_t = plt.gcf().canvas.set_window_title('Target')
    epochs['NonTarget'].plot_image(combine='mean')
    fig_nt = plt.gcf().canvas.set_window_title('Non-Target')

    return fig_t, fig_nt

def plot_erp_heatmaps_channelwise(epochs: mne.Epochs, csd_applied: bool):
    dict_electrodes = dict(eeg='EEG') if not csd_applied else dict(csd='CSD')
    for ch_type, title in dict_electrodes.items():
        layout = mne.channels.find_layout(epochs.info, ch_type=ch_type)
        fig_t = epochs['Target'].plot_topo_image(layout=layout, fig_facecolor='w',
                                         font_color='k', title=title + ' Target Trial x time amplitude')
        fig_nt = epochs['NonTarget'].plot_topo_image(layout=layout, fig_facecolor='w',
                                            font_color='k', title=title + ' Non-Target Trial x time amplitude')

        return fig_t, fig_nt

def plot_precision_recall(classifier: LinearDiscriminantAnalysis,
                          X: np.ndarray,
                          y_gt: np.ndarray):

    y_score = classifier.decision_function(X)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_gt, y_score, pos_label=classifier.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)  # .plot()

    # Precision Recall Display
    prec, recall, _ = precision_recall_curve(y_gt, y_score,
                                             pos_label=classifier.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)  # .plot()

    # Display them side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    roc_display.plot(ax=ax1)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    ax1.legend(loc="lower right")
    pr_display.plot(ax=ax2)

    plt.show()
    return fig

import mne
import numpy as np
from typing import List, Tuple, Dict, Union
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from matplotlib.ticker import MaxNLocator
from p3k import epoching

color_T = '#c10629' # Netflix red
color_NT = 'steelblue' # also very badass


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

# Group level analysis (Matthias)
# Input: List with averaged evokeds per subject
def plot_grand_average_erp(ci, evoked_per_subject, electrode, title="Grand Average"):
    title = ''.join(electrode) + " " + title
    # splitting by type to fit function signature
    evoked_T = []
    evoked_NT = []
    for i in range(evoked_per_subject.__len__()):   # [ ] check for bug: is it really iterating over different evokeds?
        evoked_T.append(evoked_per_subject[i][0])   # Target
        evoked_NT.append(evoked_per_subject[i][1])  # NonTarget

    if not ci:
        evoked_T = mne.grand_average(evoked_T, interpolate_bads=False, drop_bads=False)
        evoked_NT = mne.grand_average(evoked_NT, interpolate_bads=False, drop_bads=False)

    evokeds = dict(NonTarget=evoked_NT, Target=evoked_T)
    return mne.viz.plot_compare_evokeds(evokeds, picks=electrode, show_sensors=False,
                                              title = title, styles={"Target": {"linewidth": 3}, "NonTarget": {"linewidth": 3}},
                                              linestyles={'NonTarget': 'dashed'}, colors={'Target': color_T, 'NonTarget': color_NT})


def plot_average_erp(epochs: mne.Epochs, title=None, picks=None):
    title = ''.join(picks) + " " + title     # picks = [f'eeg{n}' for n in range(10, 15)]
    l_target, l_nt = epoching.get_avg_target_nt(epochs=epochs)
    evokeds = dict(NonTarget=l_nt, Target=l_target)
    fig_handle = mne.viz.plot_compare_evokeds(evokeds, picks=picks, show_sensors=False,
                                              title = title, linestyles={'NonTarget': 'dashed'},
                                              colors={'Target': 'r', 'NonTarget': 'b'} )
    return fig_handle

# Matthias. Like 'plot_average_erp', but without averaging of epochs to retain information necessary for CI calculation
def plot_CI_erp(epochs: mne.Epochs,
                groups: Dict[str, Union[str, List[bool]]] = {'Target': 'Target', 'NonTarget': 'NonTarget'},
                title: str = None, annotation: str = None,
                ch_subset: List[str]=None, average_channels: bool=False, display_range=None,
                figure_grid: bool = True, subplots_per_col: int = 3,
                figsize: Tuple[int,int] = None):
    title = "ERP" if title is None else title

    list_figs = []

    if ch_subset is None or len(ch_subset) == 0:
        idc_ch = range(len(epochs.ch_names))
        lbl_ch = epochs.ch_names.copy()
    elif isinstance(ch_subset, list):
        # list of indices
        if isinstance(ch_subset[0], int):
            idc_ch = ch_subset.copy()
            lbl_ch = epochs.ch_names[ch_subset]
        # list of channel names
        elif isinstance(ch_subset[0], str):
            idc_ch = [index for index, value in enumerate(epochs.ch_names) if value in ch_subset]
            lbl_ch = ch_subset
        else:
            raise NotImplementedError()

    list_loop = []
    if not average_channels:
        list_loop = [{'title': f"{title} {ch_lbl}",
                      'ch_picks': ch_idx,
                      'fn_lbl': f'erp_{ch_lbl}'} for ch_idx, ch_lbl in list(zip(idc_ch, lbl_ch))]

    else:
        ch_str = f"[{','.join(lbl_ch)}]"
        list_loop = {'title': f"{title} avg {ch_str}",
                     'ch_picks': idc_ch,
                     'fn_lbl': f'erp_AVG_{len(lbl_ch)}ch'}

    axes = None
    legend_stored = None
    # decide whether figures must be put in a grid
    if figure_grid:
        # Determine the number of rows and columns for the grid
        n = len(list_loop)
        cols = subplots_per_col if n > subplots_per_col else n
        rows = np.ceil(n / cols).astype(int)

        fig, axes = plt.subplots(nrows=rows, ncols=cols,
                            figsize=(5*cols, 5*rows) if figsize is None else figsize,
                            sharex=True, sharey=True, )

    n_drawn = -1
    for i_plot, d in enumerate(list_loop):
        it_title = d['title']
        it_picks = d['ch_picks']
        it_lbl = d['fn_lbl']

        if axes is not None:
            ax = axes[i_plot // cols, i_plot % cols]
            #ax.set_xticks([])
            #ax.set_yticks([])
        else:
            ax = None
        #if display_range == []:
        #    display_range = [-5, 10]  # make it automatic matty

        # iterate over evoked
        evoked = {}  # {'Target': epochs['Target'], 'NonTarget': epochs['NonTarget']} by default
        group_labels = []
        for group_label, epochs_filter in groups.items():
            group_labels.append(group_label)
            evoked[group_label] = list(epochs[epochs_filter].iter_evoked())

        new_fig = mne.viz.plot_compare_evokeds(evoked, picks=it_picks,
                                               axes=ax,
                                               show=~figure_grid,
                                               show_sensors=True, combine='mean',
                                               ci=True,
                                               ylim=dict(eeg=display_range) if display_range is not None else None,
                                               truncate_yaxis=False,
                                               # ci=True by default
                                               title=it_title,
                                               styles={group_labels[0]: {"linewidth": 3},
                                                       group_labels[1]: {"linewidth": 3}},
                                               linestyles={group_labels[1]: 'dashed'},
                                               colors={group_labels[0]: color_T, group_labels[1]: color_NT})


        if figure_grid:
            if legend_stored is None:
                legend_stored = ax.get_legend_handles_labels()
            ax.legend().set_visible(False)
            if not display_range:
                min_y = 0
                max_y = 0
                for ax in axes.flat:
                    cur_min = ax.dataLim.min[1]
                    cur_max = ax.dataLim.max[1]
                    cur_min = cur_min if np.isfinite(cur_min) else min_y
                    cur_max = cur_max if np.isfinite(cur_max) else max_y

                    min_y = min(min_y, cur_min)
                    max_y = max(max_y, cur_max)
                for ax in axes.flat:
                    ax.set_ylim(bottom=min_y, top=max_y)
                    ax.set_ybound(lower=min_y, upper=max_y)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


        if not figure_grid:
            list_figs.append({'ax': new_fig,
                              'title': it_title,
                              'evoked': evoked,
                              'lbl': it_lbl})


        n_drawn = max(i_plot, n_drawn)

    if figure_grid:
        # empty remaining empty subplots
        for i_empty in list(range(n_drawn+1, rows*cols)):
            ax = axes[i_empty // cols, i_empty % cols]
            ax.set_axis_off()

        if n_drawn == rows*cols:
            legend_axis = axes[-1, -1]
        else:
            legend_axis = axes[0, 0]
        legend_axis.legend(legend_stored[0], legend_stored[1], loc='upper left')
            # One could unify ylim scales by storing evoked first then making the plots

        # add an annotation in the bottom left
        if annotation is not None:
            legend_axis.annotate(annotation, xy=axes[-1, -1].viewLim.min)

        ret_fig = fig
        fig.show()
    else:

        ret_fig = list_figs


    return ret_fig




def plot_channel_average(epochs: mne.Epochs):
    nb_chans = epochs['Target']._data.shape[1]
    splt_width = int(np.floor(
        np.sqrt(1.0 * nb_chans + 2)))  # adding an extra plot with all channels combined at the end and a legend
    splt_height = splt_width if splt_width * splt_width >= nb_chans + 1 else splt_width + 1
    while splt_height * splt_width < nb_chans + 2:
        splt_height += 1
    fig, axes = plt.subplots(splt_height, splt_width, figsize=(10, 8), sharex='all', sharey='all')

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
                                         legend=False, truncate_yaxis = True, # Bugfix Matthias: added truncate_yaxis = True # 'auto' occasionally crashed when axis had only 1 tick
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
    plt.autoscale(enable=True,  axis="both", tight=None)
    plt.show()
    return fig



# Matthias  # unfinished - it runs, but shows the same signal everywhere
def plot_GA_channel_average(evoked_per_subject):
    # splitting by type to fit function signature
    evoked_T = []
    evoked_NT = []
    for i in range(evoked_per_subject.__len__()):
        evoked_T.append(evoked_per_subject[i][0])   # Target
        evoked_NT.append(evoked_per_subject[i][1])  # NonTarget
    evoked_T = mne.grand_average(evoked_T, interpolate_bads=False, drop_bads=False)
    evoked_NT = mne.grand_average(evoked_NT, interpolate_bads=False, drop_bads=False)

    evokeds = dict(NonTarget=evoked_NT, Target=evoked_T)

    #nb_chans = epochs['Target']._data.shape[1]
    nb_chans = 12
    splt_width = int(np.floor(
        np.sqrt(1.0 * nb_chans + 2)))  # adding an extra plot with all channels combined at the end and a legend
    splt_height = splt_width if splt_width * splt_width >= nb_chans + 1 else splt_width + 1
    while splt_height * splt_width < nb_chans + 2:
        splt_height += 1
    fig, axes = plt.subplots(splt_height, splt_width, figsize=(10, 8), sharex='all', sharey='all')

    nb_cells = splt_height * splt_width
    for plot_idx in range(nb_cells):
        # cells containing data
        if plot_idx in range(nb_chans):
            ch_idx = plot_idx
            print('plotting channel {}'.format(ch_idx + 1))
            mne.viz.plot_compare_evokeds(evokeds, #picks=[epochs.info['ch_names'][ch_idx]],
                                         legend=False, truncate_yaxis = True, # Bugfix Matthias: added truncate_yaxis = True # 'auto' occasionally crashed when axis had only 1 tick
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
    ax_evoked = mne.viz.plot_compare_evokeds(evokeds, combine='mean', #picks=epochs.info['ch_names'],
                                     legend=False,
                                     axes=axes[-1, -1], show=False)

    # retrieve the legend and move it in the previous cell

    plt.subplots_adjust(hspace=1, wspace=.1)
    plt.autoscale(enable=True,  axis="both", tight=None)
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
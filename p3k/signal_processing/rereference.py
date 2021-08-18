from dataclasses import dataclass
import mne
from p3k.plots import plot_seconds

def apply_infinity_reference(raw: mne.io.BaseRaw, display_plot: bool = False,
                             sphere=None, src=None, forward=None) -> mne.io.BaseRaw:

    raw.del_proj()  # remove our average reference projector first
    if sphere is None and src is None and forward is None:
        sphere = mne.make_sphere_model('auto', 'auto', raw.info)
        src = mne.setup_volume_source_space(sphere=sphere, exclude=30., pos=15.)
        forward = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
    raw_rest = raw.copy().set_eeg_reference('REST', forward=forward)

    #raw_short = ep_plot = mne.make_fixed_length_epochs(raw, duration=10)[0]
    #raw_rest_short = mne.make_fixed_length_epochs(raw_rest, duration=10)[0]

    if display_plot:
        for title, _raw in zip(['Original', 'REST (∞)'], [raw, raw_rest]):
            fig = plot_seconds(raw=_raw, seconds=10, title=title)

            #fig = raw_short.plot(n_channels=raw_short.get_data().shape[1], scalings=dict(eeg=5e-5))
            # make room for title
            fig.subplots_adjust(top=0.9)
            fig.suptitle('{} reference'.format(title), size='xx-large', weight='bold')
    else:
        fig = None
    return raw_rest, sphere, src, forward, fig

@dataclass
class P300Analysis:


    def run(self):
        pass

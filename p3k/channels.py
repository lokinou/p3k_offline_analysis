import mne
import re
from typing import List


class ChannelException(Exception):
    def __init__(self):
        super(ChannelException, self).__init__()

def flag_channels_as_bad(bad_channels: list, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    new_bads = [raw.info['ch_names'][ch] for ch in bad_channels]
    raw.info['bads'].extend(new_bads)
    raw.pick_types(eeg=True)


def define_channels(raw: mne.io.BaseRaw,
                    channel_names: List[str],
                    montage: mne.channels.DigMontage = None):
    print("Channel names from data: {}".format(raw.info['ch_names']))

    if montage is None:
        montage = mne.channels.make_standard_montage('standard_1005')

    if channel_names is None:
        channel_names = raw.info['ch_names']
        print('Using channel names directly from the data files: {}'.format(channel_names))
    elif len(channel_names) < len(raw.info['ch_names']):

        # append missing channels to cname
        channel_names.extend(raw.info['ch_names'][len(channel_names) - len(raw.info['ch_names']):])
        print('Using user defined channel names and ignoring other channels: {}'.format(channel_names))
    elif len(channel_names) > len(raw.info['ch_names']):
        print('Number of channels in data (n={}) is lower than the declared channel names {}'.format(
            len(raw.info['ch_names']), channel_names))
        raise ChannelException
    else:
        print('Using user defined channel names: {}'.format(channel_names))

    assert re.match(r'([0-9]+)', channel_names[0]) is None, \
        f'Channel names invalid: first channel="{channel_names[0]}". ' \
        f'Please define them manually in param_channels.channels'

    assert re.search(r'(channel)', channel_names[0], flags=re.IGNORECASE) is None, \
        f'Channel names invalid: first channel="{channel_names[0]}". ' \
        f'Please define them manually in param_channels.channels'


    nb_chan = len(raw.info['ch_names'])
    nb_def_ch = len(channel_names)

    # regular expressions to look for certain channel types
    eog_pattern = re.compile('eog|EOG')  # EOG electrodes
    emg_pattern = re.compile('emg|EMG')  # EMG electrodes
    mastoid_pattern = re.compile('^[aA][0-9]+')  # A1 and A2 electrodes (mastoids)

    types = []
    cname_map = dict(zip(raw.info['ch_names'], channel_names))
    for nc in channel_names:
        if eog_pattern.search(nc) is not None:
            t = 'eog'
        elif emg_pattern.search(nc) is not None:
            t = 'emg'
        elif mastoid_pattern.search(nc) is not None:
            t = 'misc'
        elif nc in montage.ch_names:  # check the 10-05 montage for all electrode names
            t = 'eeg'
        else:
            t = 'misc'  # if not found, discard the channel as misc

        types.append(t)

    type_map = dict(zip(channel_names, types))
    #print(cname_map)
    # rename and pick eeg
    raw.rename_channels(cname_map, allow_duplicates=False)
    raw.set_channel_types(type_map)
    raw.pick_types(eeg=True, misc=False)

    #print('Electrode mapping')
    raw = raw.set_montage(montage, match_case=False)

    return raw

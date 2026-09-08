from scipy import io
from scipy.signal import sosfiltfilt, iirnotch, filtfilt
import numpy as np
from .base import generate_filterbank
from sklearn.preprocessing import LabelEncoder
import resampy


def preprocess(data_file: str, config: dict):
    '''preprocess EEG data and labels of shape (N,C,T) and (N)'''
    print(f'loading data from: {data_file}')
    data = io.loadmat(data_file)

    EEG = data['EEG_data'].astype(np.float64)
    EEG = EEG.transpose((2, 0, 1))
    labels = data['labels'].reshape(-1).astype(np.int32) # N
    # label numbers must start at 0
    labels = LabelEncoder().fit_transform(labels)
    EEG = EEG[:, config['channels'], :]

    # parameter = iirnotch(50, 25, config['fs'])
    # EEG = filtfilt(parameter[0], parameter[1], EEG, axis=-1)
    fs = config['fs']
    if config['data_dur']:
        e = int((config['data_dur'][1] - config['data_dur'][0]) *fs)
        EEG = EEG[..., :e]
    EEG = resampy.resample(EEG, fs, fs//config['resample'])

    fs = fs // config['resample']

    b, a = iirnotch(50, 25, fs)
    EEG = filtfilt(b, a, EEG, axis=-1)
    # filterbank = generate_filterbank([[config['freqs'][0], 48]], [[config['freqs'][0]-2, 50]], fs, order=4, rp=1)
    # EEG = sosfiltfilt(filterbank[0], EEG, axis=-1)

    if config['win_dur']:
        s = int((config['win_dur'][0] - config['data_dur'][0]) * fs)
        e = int((config['win_dur'][1] - config['data_dur'][0]) * fs)
        EEG = EEG[..., s:e]

    EEG = EEG - np.mean(EEG, axis=-1, keepdims=True)
    print(f'preprocessing data: {EEG.shape} {labels.shape}')
    print(np.max(EEG))
    return EEG, labels
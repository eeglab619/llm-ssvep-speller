import numpy as np
import scipy.io as sio
from scipy.signal import filtfilt, iirnotch

from fbecca.FBECCA import FBECCA
from fbecca.base import generate_filterbank, generate_cca_references

# Benchmark dataset (Wang et al., 2017) recording parameters.
# Get the data from http://bci.med.tsinghua.edu.cn/ -- not included here.
BENCHMARK = {
    "path": "./data/Benchmark",          # expects S{subject}.mat per subject
    "sampling_rate": 250,
    "n_classes": 40,
    "n_blocks": 6,
    "cue_time": 0.5,
    "rest_time": 0.5,
    "shift_time": 0.14,
    "channels_idx": [60, 61, 62, 54, 55, 56, 47, 53, 57],  # O1 Oz O2 PO3 POz PO4 Pz PO5 PO6
}

# class index -> [frequency_hz, phase_in_units_of_pi], the 40 JFPM stimuli.
CLASSES = {}
_freqs, _phases = [8, 9, 10, 11, 12, 13, 14, 15], [0, 0.5, 1, 1.5]
for _i in range(40):
    _group, _j = divmod(_i, 8)
    CLASSES[_i] = [_freqs[_j] + _group * 0.2, _phases[(_j + _group) % 4]]

N_BANDS = 3
N_HARMONICS = 5


def load_subject(subject: int):
    """Returns raw EEG (40 classes, 6 blocks, 64 ch, 1500 samples)."""
    mat = sio.loadmat(f"{BENCHMARK['path']}/S{subject}.mat")["data"]  # (64,1500,40,6)
    return mat.transpose(2, 3, 0, 1).astype(np.float64)


def _window(raw, window_s, offset_s):
    """Trim to post-cue stimulus period, pick occipital channels, cut to
    [offset_s, offset_s+window_s), notch-filter line noise."""
    fs = BENCHMARK["sampling_rate"]
    start = int((BENCHMARK["cue_time"] + BENCHMARK["shift_time"]) * fs)
    end = raw.shape[-1] - int(BENCHMARK["rest_time"] * fs)
    raw = raw[..., start:end][..., BENCHMARK["channels_idx"], :]
    o, w = int(offset_s * fs), int(window_s * fs)
    raw = raw[..., o:o + w]
    b, a = iirnotch(50, 25, fs)
    return filtfilt(b, a, raw, axis=-1) - raw.mean(axis=-1, keepdims=True)


def _filterbank_and_refs(fs, n_samples):
    freqs = [CLASSES[i][0] for i in range(40)]
    phases = [CLASSES[i][1] for i in range(40)]
    T = n_samples / fs
    Yf = generate_cca_references(freqs, fs, T, phases=phases, n_harmonics=N_HARMONICS)
    f0 = min(freqs)
    wp = [[f0 * i, min(48, fs / 2 - 2)] for i in range(1, N_BANDS + 1)]
    ws = [[f0 * i - 2, min(50, fs / 2 - 1)] for i in range(1, N_BANDS + 1)]
    filterbank = generate_filterbank(wp, ws, fs, order=4, rp=1)
    filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25
    return Yf, filterbank, filterweights


def fbecca_scores_loo(windowed, fs, n_jobs=8):
    """windowed: (n_classes, n_blocks, n_ch, T) for one look. Returns
    (n_classes, n_blocks, n_classes) rho via leave-one-block-out CV: for
    each held-out block, fit per-class templates on the other blocks, then
    score the held-out block against them."""
    n_classes, n_blocks = windowed.shape[0], windowed.shape[1]
    fs_ = fs
    Yf, filterbank, filterweights = _filterbank_and_refs(fs_, windowed.shape[-1])

    all_rho = np.zeros((n_classes, n_blocks, n_classes), dtype=np.float32)
    for held_block in range(n_blocks):
        train_idx = [b for b in range(n_blocks) if b != held_block]
        train_x = windowed[:, train_idx].reshape(-1, *windowed.shape[2:])
        train_y = np.repeat(np.arange(n_classes), len(train_idx))
        test_x = windowed[:, held_block]  # (n_classes, ch, T)

        model = FBECCA(filterbank, filterweights=filterweights, n_jobs=n_jobs)
        model.fit(train_x, train_y, Yf=Yf)
        all_rho[:, held_block, :] = model.transform(test_x)
    return all_rho


def build_multilook_evidence(subject: int, n_looks: int, look_window_s: float):
    """The core signal-processing step. Returns `looks`: a list of length
    n_looks, each an (n_classes, n_blocks, n_classes) array where
    looks[k][true_class, block] is that look's FBECCA correlation score
    against every candidate class -- i.e. the raw evidence the LLM policy
    will read. Every score is honest leave-one-block-out: a trial's own
    block is never used to build the template that scores it.
    """
    raw = load_subject(subject)
    fs = BENCHMARK["sampling_rate"]

    looks = []
    for k in range(n_looks):
        windowed = _window(raw, look_window_s, offset_s=k * look_window_s)  # (cls, blk, ch, T)
        rho = fbecca_scores_loo(windowed, fs)
        looks.append(rho)
    n_classes, n_blocks = raw.shape[0], raw.shape[1]
    return looks, n_classes, n_blocks

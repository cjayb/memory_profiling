import numpy as np
from scipy.fftpack import fft, ifftshift, fftfreq
from mne import pick_types
from mne.filter import (_prep_for_filtering, _get_filter_length,
                        _filter_attenuation, _1d_overlap_filter)
from mne.fixes import get_firwin2
from mne.io import Raw
from tempfile import mkdtemp
from os.path import join as opj


def _mmap_raw(raw):
    data_shape = raw._data.shape
    tmpdir = mkdtemp(dir='/Users/cjb/tmp')
    mmap_fname = opj(tmpdir, 'raw_data.dat')

    fp = np.memmap(mmap_fname, dtype='float64', mode='w+',
                   shape=data_shape)

    fp[:] = raw._data[:]
    # delete numpy array and the memmap writer
    del raw._data
    del fp

    raw._data = np.memmap(mmap_fname, dtype='float64', mode='r+',
                          shape=data_shape)

def _do_prep(Fs, Fp, Fstop, x, copy, picks, filter_length, zero_phase):
    freq = [0, Fp, Fstop, Fs / 2]
    gain = [1, 1, 0, 0]

    firwin2 = get_firwin2()
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)

    # issue a warning if attenuation is less than this
    min_att_db = 20

    # normalize frequencies
    freq = np.array(freq) / (Fs / 2.)
    gain = np.array(gain)
    filter_length = _get_filter_length(filter_length, Fs, len_x=x.shape[1])
    N = filter_length

    if (gain[-1] == 0.0 and N % 2 == 1) \
    or (gain[-1] == 1.0 and N % 2 != 1):
    # Gain at Nyquist freq: 1: make N EVEN, 0: make N ODD
        N += 1

    # construct filter with gain resulting from forward-backward filtering
    h = firwin2(N, freq, gain, window='hann')

    att_db, att_freq = _filter_attenuation(h, freq, gain)
    att_db += 6  # the filter is applied twice (zero phase)
    if att_db < min_att_db:
        att_freq *= Fs / 2
        print('Attenuation at stop frequency %0.1fHz is only '
                '%0.1fdB. Increase filter_length for higher '
                'attenuation.' % (att_freq, att_db))

        # reconstruct filter, this time with appropriate gain for fwd-bkwd
    gain = np.sqrt(gain)
    h = firwin2(N, freq, gain, window='hann')

    n_h = len(h)
    if x.shape[1] < len(h):
        raise ValueError('Overlap add should only be used for signals '
                         'longer than the requested filter')
    n_edge = max(min(n_h, x.shape[1]) - 1, 0)

    n_x = x.shape[1] + 2 * n_edge

    min_fft = 2 * n_h - 1
    max_fft = n_x
    if max_fft >= min_fft:
        n_tot = 2 * n_x if zero_phase else n_x

        # cost function based on number of multiplications
        N = 2 ** np.arange(np.ceil(np.log2(min_fft)),
        np.ceil(np.log2(max_fft)) + 1, dtype=int)
        # if doing zero-phase, h needs to be thought of as ~ twice as long
        n_h_cost = 2 * n_h - 1 if zero_phase else n_h
        cost = (np.ceil(n_tot / (N - n_h_cost + 1).astype(np.float)) *
        N * (np.log2(N) + 1))

        # add a heuristic term to prevent too-long FFT's which are slow
        # (not predicted by mult. cost alone, 4e-5 exp. determined)
        cost += 4e-5 * N * n_tot

        n_fft = N[np.argmin(cost)]

    # Filter in frequency domain
    h_fft = fft(np.concatenate([h, np.zeros(n_fft - n_h, dtype=h.dtype)]))
    assert(len(h_fft) == n_fft)

    h_fft = (h_fft * h_fft.conj()).real
    # equivalent to convolving h(t) and h(-t) in the time domain
    return(h_fft, n_h, n_edge, orig_shape)

def mangle_x(x):
    x *= 2.

def manual_filter():
    fname='ec_rest_before_tsss_mc_rsl.fif'
    raw = Raw(fname, preload=False)
    raw.preload_data() #  data becomes numpy.float64
    _mmap_raw(raw)

    copy = False  # filter in-place
    zero_phase = True
    n_jobs = 1
    picks = pick_types(raw.info, exclude=[], meg=True)

    x = raw._data  # pointer?

    Fs = 1000.
    Fp = 20.
    Fstop = 21.
    filter_length='10s'

    print('x.shape before prepping:', x.shape)
    h_fft, n_h, n_edge, orig_shape = _do_prep(Fs, Fp, Fstop, x, copy, picks,
                                              filter_length, zero_phase)
    print('x.shape after prepping:', x.shape)

    print('Before filtering:')
    print(x[0][:10])
    for p in picks:
        mangle_x(x[p])
        # _1d_overlap_filter(x[p], h_fft, n_h, n_edge, zero_phase,
        #                           dict(use_cuda=False))
    #x.shape = orig_shape
    print('After filtering:')
    print(x[0][:10])

    print('Data type:', raw._data[0][:5])
    print(type(raw._data))

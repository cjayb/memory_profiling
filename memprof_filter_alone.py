from mne import pick_types
from mne.io import Raw
from mne.filter import (low_pass_filter, high_pass_filter, band_pass_filter,
                      notch_filter, band_stop_filter)


def filter(l_freq, h_freq, picks=None, filter_length='10s',
           l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, n_jobs=1,
           method='fft', iir_params=None, verbose=None):
    """Filter a subset of channels.
    Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
    filter to the channels selected by "picks". The data of the Raw
    object is modified inplace.
    The Raw object has to be constructed using preload=True (or string).
    l_freq and h_freq are the frequencies below which and above which,
    respectively, to filter out of the data. Thus the uses are:
        * ``l_freq < h_freq``: band-pass filter
        * ``l_freq > h_freq``: band-stop filter
        * ``l_freq is not None and h_freq is None``: high-pass filter
        * ``l_freq is None and h_freq is not None``: low-pass filter
    If n_jobs > 1, more memory is required as "len(picks) * n_times"
    additional time points need to be temporarily stored in memory.
    raw.info['lowpass'] and raw.info['highpass'] are only updated
    with picks=None.
    Parameters
    ----------
    l_freq : float | None
        Low cut-off frequency in Hz. If None the data are only low-passed.
    h_freq : float | None
        High cut-off frequency in Hz. If None the data are only
        high-passed.
    picks : array-like of int | None
        Indices of channels to filter. If None only the data (MEG/EEG)
        channels will be filtered.
    filter_length : str (Default: '10s') | int | None
        Length of the filter to use. If None or "len(x) < filter_length",
        the filter length used is len(x). Otherwise, if int, overlap-add
        filtering with a filter of the specified length in samples) is
        used (faster for long signals). If str, a human-readable time in
        units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
        to the shortest power-of-two length at least that duration.
        Not used for 'iir' filters.
    l_trans_bandwidth : float
        Width of the transition band at the low cut-off frequency in Hz
        (high pass or cutoff 1 in bandpass). Not used if 'order' is
        specified in iir_params.
    h_trans_bandwidth : float
        Width of the transition band at the high cut-off frequency in Hz
        (low pass or cutoff 2 in bandpass). Not used if 'order' is
        specified in iir_params.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fft'.
    method : str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.
    See Also
    --------
    mne.Epochs.savgol_filter
    """
    fname='ec_rest_before_tsss_mc_rsl.fif'
    raw = Raw(fname, preload=False)
    raw.preload_data() #  data becomes numpy.float64

    if verbose is None:
        verbose = raw.verbose
    fs = float(raw.info['sfreq'])
    if l_freq == 0:
        l_freq = None
    if h_freq is not None and h_freq > (fs / 2.):
        h_freq = None
    if l_freq is not None and not isinstance(l_freq, float):
        l_freq = float(l_freq)
    if h_freq is not None and not isinstance(h_freq, float):
        h_freq = float(h_freq)

    if not raw.preload:
        raise RuntimeError('Raw data needs to be preloaded to filter. Use '
                           'preload=True (or string) in the constructor.')
    if picks is None:
        if 'ICA ' in ','.join(raw.ch_names):
            pick_parameters = dict(misc=True, ref_meg=False)
        else:
            pick_parameters = dict(meg=True, eeg=True, ref_meg=False)
        picks = pick_types(raw.info, exclude=[], **pick_parameters)
        # let's be safe.
        if len(picks) < 1:
            raise RuntimeError('Could not find any valid channels for '
                               'your Raw object. Please contact the '
                               'MNE-Python developers.')

        # update info if filter is applied to all data channels,
        # and it's not a band-stop filter
        if h_freq is not None:
            if (l_freq is None or l_freq < h_freq) and \
               (raw.info["lowpass"] is None or
               h_freq < raw.info['lowpass']):
                    raw.info['lowpass'] = h_freq
        if l_freq is not None:
            if (h_freq is None or l_freq < h_freq) and \
               (raw.info["highpass"] is None or
               l_freq > raw.info['highpass']):
                    raw.info['highpass'] = l_freq
    if l_freq is None and h_freq is not None:
        low_pass_filter(raw._data, fs, h_freq,
                        filter_length=filter_length,
                        trans_bandwidth=h_trans_bandwidth, method=method,
                        iir_params=iir_params, picks=picks, n_jobs=n_jobs,
                        copy=False)
    if l_freq is not None and h_freq is None:
        high_pass_filter(raw._data, fs, l_freq,
                         filter_length=filter_length,
                         trans_bandwidth=l_trans_bandwidth, method=method,
                         iir_params=iir_params, picks=picks, n_jobs=n_jobs,
                         copy=False)
    if l_freq is not None and h_freq is not None:
        if l_freq < h_freq:
            raw._data = band_pass_filter(
                raw._data, fs, l_freq, h_freq,
                filter_length=filter_length,
                l_trans_bandwidth=l_trans_bandwidth,
                h_trans_bandwidth=h_trans_bandwidth,
                method=method, iir_params=iir_params, picks=picks,
                n_jobs=n_jobs, copy=False)
        else:
            raw._data = band_stop_filter(
                raw._data, fs, h_freq, l_freq,
                filter_length=filter_length,
                l_trans_bandwidth=h_trans_bandwidth,
                h_trans_bandwidth=l_trans_bandwidth, method=method,
                iir_params=iir_params, picks=picks, n_jobs=n_jobs,
                copy=False)

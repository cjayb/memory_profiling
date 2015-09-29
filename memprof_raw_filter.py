from mne.io import Raw

from mne import set_cache_dir, set_memmap_min_size, get_config
#fname = 'ec_rest_before_tsss_mc_rsl.fif'
# set_cache_dir('/Users/cjb/tmp/shm')
# set_memmap_min_size('100M')

print('MNE_CACHE_DIR set to', get_config('MNE_CACHE_DIR', None))
print('Arrays larger than', get_config('MNE_MEMMAP_MIN_SIZE', None),
      'are memmapped!')

def filter_raw():
    fname='ec_rest_before_tsss_mc_rsl.fif'

    raw = Raw(fname, preload=False)
    raw.preload_data() #  data becomes numpy.float64
    raw.filter(None, 40, n_jobs=4)
    del raw

    fname='ec_rest_after_tsss_mc_rsl.fif'
    raw2 = Raw(fname, preload=False)
    raw2.preload_data() #  data becomes numpy.float64
    raw2.filter(None, 40, n_jobs=4)
    del raw2

#filtered_raw = filter_raw(fname)

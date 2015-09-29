from mne.io import Raw
from mne.datasets import sample
import numpy as np
from tempfile import mkdtemp
from os.path import join as opj

def test_preload_memmap():
    tmpdir = mkdtemp(dir='/Users/cjb/tmp')
    mmap_fname = opj(tmpdir, 'raw_data.dat')

    # fname='ec_rest_before_tsss_mc_rsl.fif'
    data_path = sample.data_path(download=False)
    fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

    raw = Raw(fname, preload=False)
    # """This function actually preloads the data"""
    # data_buffer = mmap_fname
    # raw._data = raw._read_segment(data_buffer=data_buffer)[0]
    # assert len(raw._data) == raw.info['nchan']
    # raw.preload = True
    # raw.close()
    raw.preload_data(data_buffer=mmap_fname)
    data_shape = raw._data.shape

    print('Contents of raw._data after reading from fif:')
    print(type(raw._data))
    print(raw._data[100][:5])

    del raw._data
    raw._data = np.memmap(mmap_fname, dtype='float64', mode='c',
                          shape=data_shape)

    print('Contents of raw._data after RE-loading:')
    print(type(raw._data))
    print(raw._data[100][:5])

    raw.filter(None,40)
    print('Contents of raw._data after filtering:')
    print(type(raw._data))
    print(raw._data[100][:5])

    # PROBLEM: Now the filtered data are IN MEMORY, but as a memmap
    # What if I'd like to continue from here using it as an ndarray?

    del raw._data

#if __name__ == '__main__':
    # test_memmap()
#

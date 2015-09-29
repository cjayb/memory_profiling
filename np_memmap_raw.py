from mne.io import Raw
import numpy as np
from tempfile import mkdtemp
from os.path import join as opj

def test_memmap():
    fname='ec_rest_before_tsss_mc_rsl.fif'
    raw = Raw(fname, preload=False)
    raw.preload_data() #  data becomes numpy.float64
    data_shape = raw._data.shape

    tmpdir = mkdtemp(dir='/Users/cjb/tmp')
    mmap_fname = opj(tmpdir, 'raw_data.dat')

    fp = np.memmap(mmap_fname, dtype='float64', mode='w+',
                   shape=data_shape)

    print('Contents of raw._data:')
    print(raw._data[0][:10])
    print('Contents of memmap:')
    print(fp[0][:10])

    fp[:] = raw._data[:]
    print('Contents of memmap after assignment:')
    print(fp[0][:10])

    # delete numpy array and the memmap writer
    del raw._data
    del fp

    raw._data = np.memmap(mmap_fname, dtype='float64', mode='r+',
                          shape=data_shape)
    print('Contents of raw._data after loading from memmap:')
    print(raw._data[0][:10])


    raw.filter(None,40)
    print('Contents of raw._data after filtering:')
    print(raw._data[0][:10])


#if __name__ == '__main__':
    # test_memmap()
#

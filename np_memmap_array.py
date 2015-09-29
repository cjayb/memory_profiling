import numpy as np
from tempfile import mkdtemp
from os.path import join as opj
import copy

def test_memmap():

    test_array = np.random.rand(50,500000)
    data_shape = test_array.shape

    tmpdir = mkdtemp(dir='/Users/cjb/tmp')
    mmap_fname = opj(tmpdir, 'array_data.dat')

    fp = np.memmap(mmap_fname, dtype='float64', mode='w+',
                   shape=data_shape)

    print('Contents of initialised memmap:')
    print(fp[0][:5])

    fp[:] = test_array[:]
    del fp

    # This increases run time by 50%! (2.2 to 3.3 sec)
    
    # del fp  # just write it down with zeros
    #
    # for row in xrange(data_shape[0]):
    #     # fp = np.memmap(mmap_fname, dtype='float64', mode='r+',
    #     #                shape=(1, data_shape[1]),
    #     #                offset=row*data_shape[1])
    #
    #     # read in for every row
    #     fp = np.memmap(mmap_fname, dtype='float64', mode='r+',
    #                    shape=data_shape)
    #     fp[row] = test_array[row]
    #     del fp  # flush to disk for each row to avoid accumulating whole array

    # delete numpy array and the memmap writer
    del test_array

    reloaded_array = np.memmap(mmap_fname, dtype='float64', mode='r+',
                               shape=data_shape)

    print('Contents of reloaded_array after loading from memmap:')
    print(reloaded_array[0][:5])

    for row in range(reloaded_array.shape[0]):
        reloaded_array[row] *= 2.

    print('Contents of reloaded_array after doubling:')
    print(reloaded_array[0][:5])

    del reloaded_array

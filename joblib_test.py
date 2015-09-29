import numpy as np
from joblib import Parallel, delayed

parallel, p_fun, _ = parallel_func(_1d_fftmult_ext, n_jobs)
            data_new = parallel(p_fun(x[p], B, extend_x, cuda_dict)
                                for p in picks)
            for pp, p in enumerate(picks):
                x[p] = data_new[pp]

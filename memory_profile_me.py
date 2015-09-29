"""from iPython, first
>>> from memory_profile_me import function
>>> %load_ext memory_profiler
>>> %mprun -f function function()
"""

import copy
import memory_profiler

#@profile
def function():
    x = list(range(1000000))  # allocate a big list
    y = copy.deepcopy(x)
    del x
    return y

if __name__ == "__main__":
    function()

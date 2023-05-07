# from numpy.random import Generator, Philox
# rg = Generator(Philox(0))
# rnd_num = rg.uniform(0, 1, 100)
# print(rnd_num)

# add python module directory to path for import
import sys
sys.path.append('./build')
import cuPhilox as ph
arr = ph.gen_rn(1, 0, 100000)
# save the arr to txt file
import numpy as np
np.save('philox_data.npy', arr)



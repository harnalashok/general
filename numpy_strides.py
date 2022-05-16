# Last amended: 23nd Feb, 2021
# Myfolder: OneDrive/python
# Objective: Experiments in numpy array memory usage

import numpy as np
import sys

# Metadata occupies 96 bytes extra
sam = np.arange(12)
sam.itemsize                    # 4 bytes
sam.itemsize * sam.shape[0]     # 48 Size of all items
sys.getsizeof(sam)              # 144 bytes
np.int32().itemsize   # 4
np.float32().itemsize   #  4                                # So difference is 144 - 48  = 96 bytes
                                # This 96 bytes is for metadata

# Metadat occupies 96 bytes extra
# even in the case of very large array
sal = np.arange(200000)
sal.itemsize                    # 4 bytes
sal.itemsize * sal.shape[0]     # 800000 Size of all items
sys.getsizeof(sal)              # 800096 bytes


# Let us see what happens with reshaping
# Case 1
sat = sal.reshape(-1, 40000)
sat.shape      # (5, 40000)
sys.getsizeof(sat)              # 112 bytes

# Case 2
san = sam.reshape(4,3)
sys.getsizeof(san)             # 112 bytes

# What if I rehape and also copy data
cp = sal.reshape(-1,40000).copy()
sys.getsizeof(cp)     # 800112

cp1 = sam.reshape(4,3).copy()
sys.getsizeof(cp1)    # 160 (= 48 + 112)

## Views of data
v1 = sat[:2, :30]
sys.getsizeof(v1)      # 112
v2 = san[:2,:2]
sys.getsizeof(v2)      # 112

#####################

fx = np.arange(12)
fx
t = fx.reshape(4,3)
t.strides     # (12, 4)
fx.strides    # (4,)
# To move from (beginning) of t[m,n]
#   to beginning of t[m1,n1], move by:
#     (m1-m) * 12 + (n1-n) * 4

#################

np.arange(3).repeat(3)
np.arange(3).repeat([2,3,4])

###################

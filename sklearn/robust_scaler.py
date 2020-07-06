#Last amended: 6th July, 2020
# myfolder: 
# Objective: 
#            a. Illustrate how RobustScaler works
#            b. Illustrate how RobustScaler is robust
#               against outliers
#	     c. Illustrate how StandardScaler is
#  		sensitive to outliers
#
# Copy all code and %paste it in ipython. Study results.

# 1.0 Call libraries
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import iqr

# 1.1 'd' has an outlier
d = np.array([2,4,6,8,10,12,14,16,40])
d = d.reshape(-1,1)
d

# 1.2 Transform using RobustScaler
rs = RobustScaler()
print("\n1.0 RobustScaler result:\n\n", rs.fit_transform(d))
print()

# 1.3 Calculate manually:
MEDIAN = np.median(d)
IQR = iqr(d)
print("2.0 Manual calculations result\n\n", (d-MEDIAN)/IQR)   # Result same as by RobustSclaer
print("Both the above results are same.")
print("===============================")

############
print("\nNext, remove outlier and see results")
# 2.0 Remove outlier and see results
d1 = np.array([2,4,6,8,10,12,14,16])
d1 = d1.reshape(-1,1)
d1

# 2.1 Transform using RobustScaler
rs = RobustScaler()
print("\nRobustScaler result with outlier removed:\n", rs.fit_transform(d1))
print("Above RobustScaler results with & without outlier are NOT too different")
print("=======================================================================\n\n")

# Let us now try with StandardScaler
print("\n\nTry with StandardScaler\n")

ss = StandardScaler()
print("Results with outlier:\n", ss.fit_transform(d))
ss = StandardScaler()
print("\nResults without outlier:\n",ss.fit_transform(d1))
print("Differences are surely larger than that with RobustScaler")
print()
print("_____________________________________________")
print("RobustScaling is thus robust against outliers")
print("---------------------------------------------")





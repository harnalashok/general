# Last amended: 4th Dec, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\useful_code & utilities\utilities
# Ref:
# StackOverflow: https://stackoverflow.com/a/49081131/3282777
# Kaggle: https://www.kaggle.com/superant/oh-my-cat
# Wikipedia: https://en.wikipedia.org/wiki/Unary_coding
#
# Objective:
#           i)   Ordinal column encodings using Thermometer encoding
#                The coding is similar to OneHotEncoding
#                (WHEN TO USE Thermometer Encoding)
#           ii)  Create a custom-transformer
#           iii) Demonstrate Pipeline and ColumnTransformer
#                usage with custom-transformer
#


# 1.0 Call libraries
%reset -f
# 1.1 Inherit following classes
from sklearn.base import TransformerMixin, BaseEstimator
# 1.2
import numpy as np
# 1.3 Horizontally stack sparse matricies
import scipy
from scipy.sparse import hstack


# 2.0 Create custom-transformer Class.
#     Must return a numpy array or a DataFrame
#     and NOT any other object, say, a list

class ThermometerEncoder(TransformerMixin, BaseEstimator):
    """
    Assume no NaN value

    """
    def __init__(self, sort_dict):
        self.sort_dict = sort_dict


    # 2.1 All computation should occur in fit, and if fit
    #     needs to store the result of a computation, it should
    #      do so in an attribute with a trailing underscore (_).
    #       Ref: https://dbader.org/blog/meaning-of-underscores-in-python
    def fit(self, X, y=None):
        return self

    # 2.2 Transform method
    def transform(self, X, y=None):
        thermos = []
        for i in X.columns:
            if i in self.sort_dict.keys():
                # 2.3 See explantions below, line-by-line
                col_dict_ = self.sort_dict[i]
                val_ = X[i].map(col_dict_)
                length_ = len(set(val_))
                result_ = scipy.sparse.coo_matrix(np.arange(length_) < np.array(val_).reshape(-1, 1)).astype(int)
                thermos.append(result_)
        result_ = hstack((thermos[0],thermos[1]))
        # 2.4 Return a numpy array or a DataFrame
        return result_


########### Class usage ##########

# 3.1 Call libraries
import pandas as pd
import numpy as np

# 3.2 Create a pandas DataFrame:
data = [('small', 23, 30,'a', 'manager'),('middle', 34,56,'a','DyManager'),('large', 33,67,'b','SrManager'),
        ('small',90,76,'b','SrManager'), ('small', 12, 21,'b','DyManager'),('large',34,56,'c','SrManager'),
        ('large', 89,90,'c','manager'),('middle', 75,32,'a','manager'), ('large',88, 77,'c','manager'),
        ('small', 23,32,'a','DyManager'), ('middle', 11,22,'b','SrManager'), ('large', 66,22,'d','SrManager'),
        ('large', 30,13,'d','manager'), ('middle', 11,22,'b','DyManager'), ('small', 66,22,'d','manager')]

# 3.2.1
df = pd.DataFrame.from_records(data, columns = ['x1','x2','x3','x4','x5'])
df

# 4.0 There are two ordinal columns ['x1' , 'x4']
#      For each one of the ordinal columns, levels
#       may map to digits as per rank of levels, as:
levels_map = {
               'x1' : {'small': 0, 'middle' : 1, 'large' : 2},
               'x4' :  { 'a' : 0, 'b' : 1 , 'c': 2,'d': 3 }
             }

# 4.1
# Create an instance of ThermometerEncoder
enc = ThermometerEncoder(levels_map)
# 4.1.1 fit() it
enc.fit(df)
# 4.1.2 transform() data now
out = enc.transform(df)
out
# 4.1.3
out = enc.fit_transform(df)

# 4.2
type(out)    # scipy.sparse.coo.coo_matrix
# 4.3 sparse-to-dense
out.toarray()

#  Check

# 5.0 Transform the rest of columns to sparse matrix
#     coo_matrix: COOrdinate sparse matrix
rest_columns = ['x2', 'x3']
t = scipy.sparse.coo_matrix(df[rest_columns])
t

# 5.1 Stack transformed-ordinals and remaining columns
s = hstack((out,t))

# 5.2 Result
s.toarray()
df

############# Pipeline ##########################
# 6.0
from sklearn.pipeline import Pipeline
# 6.1
pipe = Pipeline([('px', ThermometerEncoder(levels_map)) ])
pipe.fit(df)
pipe.transform(df)

############# Column Transformer ###############################
# 7.0
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# 7.1
ct = ColumnTransformer(
                        [
                          ('ct',ThermometerEncoder(levels_map), ['x1','x4'] ),
                          ('ohe', OneHotEncoder(), ['x5']),
                          ('ss', StandardScaler(), ['x2','x3'])
                          ])

# 7.2
ct.fit(df)
final = ct.fit_transform(df)
final.shape     # (15, 12 = (3+4)+2 (num)+3 (ohe))
final

#############################################
# 8.0 line-by-line explanations of tranform method
X = df
X['x1']

# 8.1
col_dict_ = {'small': 0, 'middle' : 1, 'large' : 2}
# 8.2
val_ = X['x1'].map(col_dict_)
val_
# 8.3
length_ = len(set(val_))
length_     # 3

# 8.4
np.arange(length_)
np.array(val_).reshape(-1,1)
np.arange(length_) == np.array(val_).reshape(-1,1)
np.array([0,1,2])  < np.array([[0],[1],[2]])
(np.array([0,1,2])  < np.array([[0],[1],[2]])).astype(int)

# 8.5
scipy.sparse.coo_matrix((np.array([0,1,2])  < np.array([[0],[1],[2]])).astype(int))
scipy.sparse.coo_matrix((np.array([0,1,2])  < np.array([[0],[1],[2]])).astype(int)).toarray()

################################
WHEN TO USE THERMOMETER ENCODING
################################
Use Thermomter encoding when the levels in ordinal feature
follow a pattern. For example, suppose, levels are:
low, high, veryHigh, extraHigh
Then use Thermometer encoding, if:
P(high) > P(low)
P(veryHigh) > P(high) + P(low)
P(extraHigh) > P(veryHigh) + P(high)
Or, when:
P(n) = 2^(-n)
ie
P(level1) = 1/2
P(level2) = 1/4
P(level3) = 1/8 ...and so on
See Wikipedia: https://en.wikipedia.org/wiki/Unary_coding
##################################

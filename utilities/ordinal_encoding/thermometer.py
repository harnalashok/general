# Last amended: 2nd Dec, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\useful_code & utilities\utilities
# Ref:
# StackOverflow: https://stackoverflow.com/a/49081131/3282777
# Kaggle: https://www.kaggle.com/superant/oh-my-cat
# Wikipedia: https://en.wikipedia.org/wiki/Unary_coding
#
# Objective:
#           i) Ordinal column encodings using Thermometer encoding
#              The coding is similar to OneHotEncoding
#
# Note: There  are two ways to code. Both are below.
#       SEE VER1FOR MORE DETAILS


# 1.0 Call libraries
from sklearn.base import TransformerMixin
from itertools import repeat
import scipy

# 2.0 Class that will perform encoding
#     Output is a sparse matrix
#     Takes, one data column at a time
class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        #print("self.sort_key", self.sort_key)
        self.value_map_ = None

    def fit(self, X, y=None):
        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        #print("self.value_map_:=>", self.value_map_)
        return self

    def transform(self, X, y=None):
        values = X.map(self.value_map_)

        possible_values = sorted(self.value_map_.values())

        idx1 = []
        idx2 = []

        all_indices = np.arange(len(X))

        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))
        # This 'return' is faulty for use of class
        # in ColumnTransformer. See ver 1
        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
        return result

# 3.0 Usage of Class

# 3.1 Call libraries
import pandas as pd
import numpy as np

# 3.2 Create a pandas DataFrame:
data = [('small', 23, 30,'a'),('middle', 34,56,'a'),('large', 33,67,'b'),
        ('small',90,76,'b'),  ('small', 12, 21,'b'),('large',34,56,'c'),
        ('large', 89,90,'c'),('middle', 75,32,'a'), ('large',88, 77,'c'),
        ('small', 23,32,'a'), ('middle', 11,22,'b'), ('large', 66,22,'d'),
        ('large', 30,13,'d'), ('middle', 11,22,'b'), ('small', 66,22,'d')]
# 3.2.1
df = pd.DataFrame.from_records(data, columns = ['x1','x2','x3','x4'])
df

# 4.0 Start conversion:
ordinal_columns = ['x1','x4']

thermos =[]
# 4.1
for i in ordinal_columns:
    if i == ordinal_columns[0]:
        # Create a function to use in sorted
        #   Arrange the list in the order
        #    ordinal-values increasing importance
        sort_key = ['small', 'middle', 'large'].index
    elif i == ordinal_columns[1]:
        sort_key = ['a', 'b', 'c','d'].index
    else:
         raise ValueError(i)
    #sort_key = list(set(df[i])).index
    enc = ThermometerEncoder(sort_key)
    X = enc.fit_transform(df[i])
    thermos.append(X)


# 4.2
type(thermos[0])    # scipy.sparse.coo.coo_matrix
# 4.3 Check
thermos[0].toarray()
thermos[1].toarray()

# 5.0 Transform the rest of columns to sparse matrix
#     coo_matrix: COOrdinate sparse matrix
rest_columns = ['x2', 'x3']
t = scipy.sparse.coo_matrix(df[rest_columns])
t = list([t])
t[0].toarray()

# 6.0 Stack transformed-ordinals and remaining columns
s = scipy.sparse.hstack( t + thermos ).tocsr()
# 6.1 Result
s.toarray()
##################
# Another way
##################

# 7.0 Call libraries
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import scipy


# 7.1
class ThermometerEncoder(TransformerMixin, BaseEstimator):
    """
    Assume on NaN value

    """
    def __init__(self, sort_dict):
        self.sort_dict = sort_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        thermos = []
        for i in X.columns:
            if i in self.sort_dict.keys():
                col_dict = self.sort_dict[i]
                val = X[i].map(col_dict)
                length = len(set(val))
                result = scipy.sparse.coo_matrix(np.arange(length) < np.array(val).reshape(-1, 1)).astype(int)
                thermos.append(result)
        # This 'return' is faulty for use of class
        # in ColumnTransformer. See ver 1
        return thermos


# 7.2 There are two ordinal columns ['x1' , 'x4']
#      For each one of the ordinal columns, levels
#       may map to digits as per rank of levels, as:
levels_map = {
               'x1' : {'small': 0, 'middle' : 1, 'large' : 2},
               'x4' :  { 'a' : 0, 'b' : 1 , 'c': 2,'d': 3 }
             }

# 7.3
enc = ThermometerEncoder(levels_map)
enc.fit(df)
out = enc.transform(df)
out
out = enc.fit_transform(df)

# 7.4
out[0].toarray()
out[1].toarray()

# 7.5
type(out[0])    # scipy.sparse.coo.coo_matrix
# 7.6 Check
out[0].toarray()
out[1].toarray()

# 7.7 Transform the rest of columns to sparse matrix
#     coo_matrix: COOrdinate sparse matrix
rest_columns = ['x2', 'x3']
t = scipy.sparse.coo_matrix(df[rest_columns])
t = list([t])
t[0].toarray()

# 7.8 Stack transformed-ordinals and remaining columns
s = scipy.sparse.hstack( t + out ).tocsr()
# 7.9 Result
s.toarray()
df
#######################################
from sklearn.pipeline import Pipeline

pipe = Pipeline([('px', ThermometerEncoder(levels_map)) ])
pipe.fit(df)
pipe.transform(df)

#######################################

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('ct',ThermometerEncoder(levels_map), ['x1','x4'] )])
ct.fit(df)

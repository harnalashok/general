# Last amended: 4th Dec, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\useful_code_utilities\utilities
# See file: 'thermometer.py' in this folder for full
#           explanation of class
#
#
# Objective:
#           i) This file acts as a python module for the class
#              'ThermometerEncoder'
#

# 1.1 Inherit following classes
from sklearn.base import TransformerMixin, BaseEstimator
# 1.2
import numpy
# 1.3 Horizontally stack sparse matricies
import scipy
from scipy.sparse import hstack


# 2.0 Create custom-transformerclass:
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
                col_dict_ = self.sort_dict[i]
                val_ = X[i].map(col_dict_)
                length_ = len(set(val_))
                result_ = scipy.sparse.coo_matrix(numpy.arange(length_) < numpy.array(val_).reshape(-1, 1)).astype(int)
                thermos.append(result_)
        result_ = hstack((thermos[0],thermos[1]))
        return result_



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        If NaN exist fill them up first
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

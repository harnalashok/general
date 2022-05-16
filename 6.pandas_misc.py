# Last amedned: 15th Jan, 2020
# My folder: /home/ashok/Documents/1.basic_lessons
# Objectives:
#           i)  Concatenating datasets in pandas
#           ii) Missing values treatment
#
# 1.0
import pandas as  pd
import numpy as np

# 1.1 Create three dataframes with same indexes
xx = pd.DataFrame(np.random.rand(10,4), columns = list('abcd'), index = range(10))
yy = pd.DataFrame(np.random.randn(5,4), columns = list('abcd'), index = range(10))
zz = pd.DataFrame(np.random.uniform(low=20,high=40,size = (10,4)), columns = list('abcd'), index = range(10))
xx
yy
zz

# 1.2 Vertical stacking
a1 = pd.concat([xx,yy,zz])        # Stacking. row nos repeat
a1
a1.loc[1,]                        # Row with index =1. Three rows

# 1.3
a2 = pd.concat([xx,yy,zz], ignore_index=True) # Concat and reindex
a2
a2.loc[1,]       # Only one row now


# 2.0 Column-wise concatenate
a3 = pd.concat([xx,yy,zz], axis = 'columns')      # Concat horizontally
a3                        # column names repeat
a3 = pd.concat([xx,yy,zz], axis = 'columns', ignore_index=True)      # column names are reassigned
a3
a3.columns = list('abcdefghijkl')
a3

# 2.1 Create a series and concatenate with xx horizontaly
s = pd.Series(['a', 'b'] * 5)
s
pd.concat([xx,s], axis = 1)
pd.concat([xx, pd.DataFrame(s)], axis =1)

# 3. Create a dataframe with different indexes
xx = pd.DataFrame(np.random.rand(10,4), columns = list('abcd'), index = range(10, 20,1))
yy = pd.DataFrame(np.random.randn(5,4), columns = list('abcd'), index = range(30, 35,1))
zz = pd.DataFrame(np.random.uniform(low=20,high=40,size = (10,4)), columns = list('abcd'), index = range(100,110))
xx
yy
zz

# 3.1 Start Concatenating
pd.concat([xx,yy,zz])                       # Works
pd.concat([xx,yy,zz], ignore_index=True)    # Works
pd.concat([xx,yy,zz], axis =1)              # Generates NaN
pd.concat([xx,yy,zz], axis =1, ignore_index=True)         # Generates NaN
xx.reset_index()    # We have an added column of index
xx.reset_index(drop = True)   # Additional column is dropped
# 3.2 This is good
pd.concat([xx.reset_index(drop = True),yy.reset_index(drop = True),zz.reset_index(drop = True)], ignore_index=True, axis =1)           # Works

################
# Missing data
################

# 4.0 Create data with missing values
df = pd.DataFrame(np.random.randn(10,4) , columns = list('abcd'))
df
df.iloc[5:7,1] = None
df.iloc[6,1:3] = None
df

# 4.1 Count column wise no of nulls
df.isnull().apply(np.sum)
# 4.2 Count row-wise nulls
df.isnull().apply(np.sum, axis = 1)
# 4.3 Count total nulls
np.sum(np.sum(df.isnull()))

# 4.4 Use apply function to alculate
#     column-wise mean or median
df.apply(np.mean)

# 4.5
dd= df.apply(np.mean)
# 4.6 One way to fill na
#       NA in each column is replaced by column mean
df.fillna(dd)
# 4.6 Transform Series output to dictionaery
dd = dict(df.apply(np.mean))
dd

# 4.7 Use dictinary values to fill NAs
#     column names are dictionary keys
df.fillna(dd)
# 5.0 Simply put an indicator value wherever misisng data is:
df.fillna(9999)

# 6.0  Replace numeric column with mode
from scipy.stats import mode
df = pd.DataFrame({ 'A' : ['abc', 'cde', 'abc', 'abc',None] * 2, 'B' : [1,2,1, None, 1] * 2})
df
# 6.1 Calculate mode
k = df[['B']].apply(mode)
k[0][1]
# 6.2 As mode is 1D array, make it scalar and fill na
df.fillna({'B': np.asscalar(k[0][1])})

###########################

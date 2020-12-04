# Last amended: 4th December, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\useful_code_utilities\utilities
#
# Objective: Demo of Pipelining in a custom transformer
#

%reset -f
# 1.0
# 1.3
import pandas as pd
import numpy
# 1.4
from sklearn.pipeline import Pipeline
import os

# 1.1 Where is our 'module' file 'ordinal.py' ?
os.chdir("C:\\Users\\Administrator\\OneDrive\\Documents\\useful_code_utilities\\utilities")

# 1.2 Call 'module', 'ordinal'
from  ordinal  import ThermometerEncoder


# 2.0
data = [('small', 23, 30,'a', 'manager'),('middle', 34,56,'a','DyManager'),('large', 33,67,'b','SrManager'),
        ('small',90,76,'b','SrManager'), ('small', 12, 21,'b','DyManager'),('large',34,56,'c','SrManager'),
        ('large', 89,90,'c','manager'),('middle', 75,32,'a','manager'), ('large',88, 77,'c','manager'),
        ('small', 23,32,'a','DyManager'), ('middle', 11,22,'b','SrManager'), ('large', 66,22,'d','SrManager'),
        ('large', 30,13,'d','manager'), ('middle', 11,22,'b','DyManager'), ('small', 66,22,'d','manager')]


df = pd.DataFrame.from_records(data, columns = ['x1','x2','x3','x4','x5'])
df

levels_map = {
               'x1' : {'small': 0, 'middle' : 1, 'large' : 2},
               'x4' :  { 'a' : 0, 'b' : 1 , 'c': 2,'d': 3 }
             }


############# Pipeline ##########################
from sklearn.pipeline import Pipeline
pipe = Pipeline([('px', ThermometerEncoder(levels_map)) ])
pipe.fit(df)
pipe.transform(df)

############# Column Transformer ###############################

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

ct = ColumnTransformer(
                        [
                          ('ct',ThermometerEncoder(levels_map), ['x1','x4'] ),
                          ('ohe', OneHotEncoder(), ['x5']),
                          ('ss', StandardScaler(), ['x2','x3'])
                          ])

ct.fit(df)
final = ct.fit_transform(df)
final.shape     # (15, 12 = (3+4)+2 (num)+3 (ohe))
final

########################
import ordinal
ordinal

"""
Last amended: 8th Feb, 2021
Myfolder:  /home/ashok/Documents/1.basic_lessons


Reference: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#series
           https://docs.python.org/2/tutorial/introduction.html#unicode-strings


Objectives:
	i)  Data structures in pandas: Series, DataFrame and Index
	ii) Data structures usage


"""

import pandas as pd
import numpy as np
import os


########Series#############
## A. Creating Series
# 10. i) Series is an array
#    ii) It is one-dimensional.
#   iii) It is labeled by index or labels
#    iv) Is dtype may be numeric or object
#         or 'category'


# 10.1 Exercises
s = pd.Series([2,4,8,10,55])
s
type(s)
s.name = "AA"
s


# 10.2 This is also a series but stores list objects
t = pd.Series({'a' : [1,2,3,4,], 'b' : [5,6]})
t
type(t)



# 10.3 Exercise
ss=[23,45,56]
h=pd.Series(ss)
h

# 10.4 OR generate it as:
h=pd.Series(range(23,30,2))
h


## B. Simple Operations
# 10.5 Exercise
s+h
s*h
s-h

(s+h)[1]       # Note the indexing starts from 0
s*h[2]


s.mean()
s.std()
s.median()


## C. Series as ndarray
 # 10.6 Also series behaves as ndarray
 #      Series acts very similarly to a ndarray,
 #      and is a valid argument to most NumPy functions.

np.mean(s)
np.median(s)
np.std(s)


## D. Indexing in series
# 10.7 Exercise
d=pd.Series([4,5], index=['a','b'])
e=pd.Series([6,7], index=['f','g'])
f=pd.Series([9,10], index=['a','b'])
d+e  # All NaN
d+f


# 10.8 Reset index of 'd' and check
v = d.reset_index()
type(v)            # v is a DataFrame


# 10.9
d.reset_index(
              drop = True,     # drop = False, adds existing index as
              inplace = True   # a new column and makes it a DataFrame
              )

d

e.reset_index(drop = True, inplace = True)
d + e


## E. Accessing Series
# 10.10 Exercise
#       Create the following series and access
#       a) elements > 0, b) elements > series-mean
j= pd.Series(np.random.normal(size=7))


k=j[j>0]
k=j[j>np.mean(j)]
k


# 10.11 Exercise
#       Create the following series
#       a) access its first two elements
#       b) acccess elements from index 2 to 4
#       c) get mean of elements from index 2 to 4
k = pd.Series(
             np.random.normal(size=7),
             index=['a','b','c','d','e','f','a']
             )


k['a']   # 'a' is duplicate index
k.loc['a']
k[:2]    # Show first two or upto 2nd index (0 and 1)
k.iloc[:2]

# 10.12
k.iloc[2:]    # Start from 2nd index
k.iloc[2:4]   # Start from IInd index upto 4th index
k.iloc[2:4].mean()


# 10.13  SURPRISE HERE!
#        a) Access first two elements
#        b) explain diff between k[2], k.loc[2], k.iloc[2]
k = pd.Series(np.random.normal(size=7),index=[0,2,5,3,4,1,6])



k.loc[0]                  # Access by index-name
k.loc[1]                  # Access by index-name
k.iloc[:2]                # Access by position
k.iloc[[0,1,2]]           # Access by index-name
k.take([0,1,2])           # Access by position
k.loc[[0,1,2]]


# 10.14 Exercise
# A series is like a dictionary. Can be accessed by its index (key)
#  Access all values from 'b' to 'f'
#  Assign ll values from 'b' to 'f' with 5.0
e=pd.Series(np.random.uniform(0,5,7), index=['a','b','c','d','e','f','g'])
e


e['a' : 'e']
e.loc['a' : 'e']

e['a' : 'd']   # All values from 'a' to 'd'
e['b' : 'd']
e.take(['b' : 'd'])
e+k
e['b' : 'f'] = 5.0


# 10.15 Compare memory usage by category type
#        Internally categories store data as integers
#        plus some meta-data that point to actual mapping
#   	 of integers
#  Ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html

f = [1,2] * 500
r = pd.Series(f)
r.memory_usage()	# 8128
e = r.astype('category')
e.memory_usage()	# 1224

# 10.15.1
r.dtype			# int64
t = r.astype('int8')    # int8 varies from -127 to +128
t.memory_usage()	# 1128 + metadata = 1224
r.values.itemsize
t.values.itemsize

######## DataFrame ###########
"""
###
Questions answered
1. Create dataframe:
    a. From Pandas Series
    b. From dictionary
    c. From Records
    d. From numpy arrays

2. Read following data, as:

    path = "C:\\Users\\Administrator\\OneDrive\\Documents\\python\\basic_lessons"
    data=pd.read_csv("delhi_weather_data.zip")
    and perform the following questions

    a. Show column-wise statistical summary
    b. Convert datetime_utc field to datetime
    c. Extract 'month', 'day' from datetime using accessor object
    d. List distinct values in field: '_conds'
    e. How many distinct values exist in '_conds'
    f. Extract rows 3 to 5 and cols 2 to 5
    g. Extract rows 3,5,7 and cols 2,5
    h. Count total number of nulls
    i.Display column wise nulls and sort them in descending order
    j.Convert ' _conds' to category type
    k.Calculate memory reduction in data[' _conds'] after
        transformation to 'category' type
    l. Transform all categoris in ' _conds' to integers
    m. From data, select only those columns with data type 'float64'
    n. Calculate row-wise mean of first four rows
    o. Calculate column-wise mean of first four columns

3. In numpy find out range or limits of following data types:
   'float64', 'float16', 'int32'

"""




'''
DataFrame is a 2-dimensional labeled data structure with columns
of potentially different types. You can think of it like a spreadsheet
or SQL table, or a dict of Series objects. It is generally the most
commonly used pandas object. Like Series, DataFrame accepts many
different kinds of input.
'''
# 1.0 Creating Dataframes

	1.1 From pandas Series
	s1 = pd.Series([1,2,3])
	s2 = pd.DataFrame(s1, columns = ['A'] )
	type(s2)

	# 1.2 From dictionary:

		# 1.2.1 Dictionary of list
		s3 = pd.DataFrame({ 'a' : [1,2,3] , 'b' : [3,4,5] })
		s3
		# 1.2.2 Dictionary of Series objects
		s4 = pd.DataFrame({ 'a' : pd.Series([1,2,3]),'b' : pd.Series([4,5,6]) })
		s4

	# 1.3 From numpy array
	s5 = pd.DataFrame(np.random.normal(size = (5,2)), columns = ['A', 'B'])
	s5

	# 1.4 From list of lists
	s6 = pd.DataFrame([[1,2,3], [3,4,5], [5,5,7]], columns = ['x', 'y', 'z'])
	s6

    # 1.5 from an array or list or tuple of records:
            rec  = [(23,25,27,"abc"), (29,54,73,"cde")]
            rec1 = ((23,25,27,"abc"), (29,54,73,"cde"))
            pd.DataFrame.from_records(rec)
            pd.DataFrame.from_records(rec,columns = ['a','b','c','d'])
            pd.DataFrame.from_records(rec1,columns = ['a','b','c','d'])

# 2
import os
path = "/home/ashok/datasets/delhi_weather"
path = "D:\\data\\OneDrive\\Documents\\python\\basic_lessons"
path = "C:\\Users\\Administrator\\OneDrive\\Documents\\python\\basic_lessons"
# 2.1
os.getcwd()
# 3
os.chdir (path)
# 4
data=pd.read_csv("delhi_weather_data.zip")
type(data)
# 5.1
pd.options.display.max_columns = 200
# 5.2
data.head()
data.tail()
data.dtypes
data.shape		# (100990, 20)
data.columns
data.values
data.columns.values
data.describe()

# 6. Datetime conversions
# Ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components
pd.to_datetime(data['datetime_utc'])
data['datetime'] = pd.to_datetime(data['datetime_utc'])
data.head()

# 6.1 using accessor 'dt':
"""
What is an accessor method?
Ref: https://www.geeksforgeeks.org/accessor-and-mutator-methods-in-python/
So, you all must be acquainted with the fact that the internal
data of an object must be kept private. But there should be
some methods in the class interface that can permit the user
of an object to access and alter the data (that is stored internally)
in a calm manner. Therefore, for that case we have two methods namely
Accessors (or 'get' methods) and Mutators (or 'set' methods) that are
helpful in accessing and altering respectively, internally stored data.
Accessor Method: This method is used to access the state of the object
                 i.e, the data hidden in the object can be accessed from
                 this method. However, this method cannot change the state
                 of the object, it can only access the data hidden. We can
                 name these methods with the word get.


"""
data['month'] = data['datetime'].dt.month
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['weekday'] = data['datetime'].dt.weekday
data['hour'] = data['datetime'].dt.hour
data['week'] = data['datetime'].dt.week
data.head(2)

# 6.2
pd.unique(data['_conds'])	# Unique values. Try with numpy?
				#  First .fillna("abc") & then try
data['_conds'].nunique()        # 39
data['_conds'].value_counts().sort_values(ascending = False)
data.head()

# 7.0 Integer Selection
data.columns
data.iloc[3:5, 1:2]  # 3:5 implies start from 3rd pos uptil 5-1=4th pos
data.iloc[3:5, 1:3]  # Display column numbers 2nd and 3rd
data.iloc[3:5, :]		# Display all columns
data.iloc[: ,3:5]		# Display all rows
data.iloc[1,1]        # Same as df[1,1:2]. Treat 1 as lower bound

# 7.1 Fancy indexing in pandas
data.iloc[[3,5,7],[1,3]]		# Specific rows and columns

# 7.1.1 Try using numpy:
data.values[[3,5,7],[1,3]]		# It fails, why?

# 7.2 Row-wise filteration using boolean indexing
data[data.month == 10 ].head()
data[(data.month == 10) & (data["_conds"] == 'Smoke') ].head()
data[(data._conds == 'Smoke') | (data._wdire == 'East')]


# 8.0 Overall how many values are nulls
np.sum(data.isnull()).sort_values(ascending = False)


# 9.0 Converting categorical variables to numeric
#     sklearn's labelencoder is one way to do it
#     Two step process:
#                1st. Convert dtype from 'object' to 'category'
#                2nd. Get integer-codes behind each category/level
#                3rd. Get correspondence behind category and integer

data['_conds'] = data['_conds'].astype('category')  # Convert to categorical variable
data['int_conds']=data['_conds'].cat.codes          # Create a column of integer coded categories
x = data[['_conds', 'int_conds']].values            # Get dataframe as an array
out = set([tuple(i) for i in x])                    # Get unique tuples of (code,category)



# 10.0 Memory reduction by changing datatypes
data.dtypes

# 10.1 Select data subset with dtype as 'float64'
newdata = data.select_dtypes('float64')

# 10.2 What are max and min data values
np.min(np.min(newdata))        # -9999
np.max(np.max(newdata))        # 101061443.0

# 10.3 What are the limits of various float datatypes
np.finfo('float64')    # finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)
np.finfo('float32')    # finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)
np.finfo('float16')    # finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16)
np.iinfo('int64')
np.iinfo('int16')

# 10.4 Change all columns to float32
# 10.4.1 What is the present memory usage
np.sum(newdata.memory_usage())             # 8887200
# 10.4.2 Change data type now
for col in newdata.columns.values:
    newdata[col] = newdata[col].astype('float32')
# 10.4.3 What is the current datausage
np.sum(newdata.memory_usage())             # 4443640 (around 50% reduction)

###############################################################################

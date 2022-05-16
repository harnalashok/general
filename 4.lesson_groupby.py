"""
Last amended: 8th Feb, 2021

Objective:
	Simple experiments using pandas groupby

What is GroupBy?
	(Ref: https://pandas.pydata.org/docs/user_guide/groupby.html)

	By “group by” we are referring to a process
	involving one or more of the following steps:

	    i)   Splitting the data into groups based
		 on some criteria.
	    ii)  Applying a function to each group
		 independently.
	    iii) Combining the results into a data
		 structure.

	Out of these, the split step is the most
	straightforward. In fact, in many situations
	we may wish to split the data set into groups
	and do something with those groups. In the
	apply step, we might wish to one of the following:

    	i)   Aggregation: compute a summary statistic
	         (or statistics) for each group.
	   ii)   Transformation: perform some group-specific
	         computations and return a like-indexed object.
	  iii)   Filtration: discard some groups, according
	         to a group-wise computation that evaluates
	         True or False.



"""

# 1.0 Call libraries
import pandas as pd
import numpy as np

# 2.0 Define a simple dataframe
#     Specify a list of tuples. Each tuple
#     constitutes a row of dataframe.
#     column (headings) and row-names are to be
#     specified separately.
#     You can create it in two ways:

df = pd.DataFrame([
                     ('bird', 'Falconiformes', 389.0, 21.2),     # row 1
					 ('bird', 'peacock',       38.0, np.nan),
                     ('bird', 'peacock',       np.nan, 23.5),      # row 2
					  ('bird', 'peacock',      35, 23.5),      # row 2
                     ('mammal', 'Carnivora',   80.2, 29.0),      # row 3
                     ('mammal', 'Primates',   np.nan, 30.6),      # row 4
                     ('mammal', 'Carnivora',   58,   np.nan),
                     ('fish', 'Whale',         89,  120.8),
                     ('fish', 'Shark',         78,   80.8),
					 ('fish', 'dolphin',      278,   np.nan),
                  ],
                  index =   ['falcon', 'p1','p2','p3', 'lion', 'monkey', 'leopard','whale','shark','dolphin'],
                  columns = ('class', 'order', 'max_speed', 'max_wt'))


# OR like:

df = pd.DataFrame.from_records([
                     ('bird', 'Falconiformes', 389.0, 21.2),     # row 1
					 ('bird', 'peacock',       38.0, np.nan),
                     ('bird', 'peacock',       np.nan, 23.5),      # row 2
					  ('bird', 'peacock',      35, 23.5),      # row 2
                     ('mammal', 'Carnivora',   80.2, 29.0),      # row 3
                     ('mammal', 'Primates',   np.nan, 30.6),      # row 4
                     ('mammal', 'Carnivora',   58,   np.nan),
                     ('fish', 'Whale',         89,  120.8),
                     ('fish', 'Shark',         78,   80.8),
					 ('fish', 'dolphin',      278,   np.nan),
                  ],
                  index =   ['falcon', 'p1','p2','p3', 'lion', 'monkey', 'leopard','whale','shark','dolphin'],
                  columns = ('class', 'order', 'max_speed', 'max_wt'))


df

"""
Questions:
1.Calculate
  For whole of data frame:
  	a. Find min,max,std of DataFrame by axis
	b. Find min, max, std of DataFrame for whole data
	c. Find min of 'max_wt' but max of 'max_speed'
	d. Square 'max_wt' and then take mean

2.Group by 'class' and 'order'
	Find number of items in each sub-group
	Display each subgroup
	For each sub-group, calculate mean, median, sum of 'max_speed'
	Count items in each sub-group
	Count items in one particular sub-group
	Sub-group wise ('class' wise) fill up nan by sub-group mean
	Show column statistics for each sub-group
	For each sub-group perform sum and mean at the same time
	For each sub-group, apply different operations to 'max_speed' and 'max_wt'

3.
	For each sub-group create a column that displays count, mean, median
	For each sub-group square up 'max_wt' and take its mean

"""

gr.describe()


def xyz(d):
	return d.fillna(d.mean())

gr = df.groupby(df['class'])

gr.agg({'max_wt': np.mean , 'max_speed' : np.median})


gr.apply(xyz)

df.fillna(df.mean())


help(gr)













# 2.1 DataFrame level aggregations:
#     These methods apply series wise
#     DataFrame methods are: sum(),
#     min(), max(), mean(), std()
#     count(), agg() and apply()

df['max_speed'].sum()
df['max_speed'].agg([np.sum, np.mean])
df.agg({'max_speed': np.sum,
        'max_wt': np.std })

# 2.2 for whole dataframe
df.mean(axis = 0)	    # Across rows
df.mean(axis = 1)     	# Across columns

# 2.3 Using apply to use an arbitrary function
#     apply() can only be passed a pandas Series

def fox(x):
    x = x * x
    return x    # Return can be a scalar or a pandas Series

df['max_speed'].apply(fox)

# 2.3.1 But this will now work:

def fox(d):
    return d['max_speed']  * d['max_speed']

df.apply(fox)

# The reason is apply(), applies the function iteratively
#  ie one by one to each series of dataframe. And, d['max_speed']
#    will not be available in every series.
#  See this: https://stackoverflow.com/a/54432584/3282777


################
## 3. Splitting
#############

# 3.1 Various ways to groupby
#      Grouping is by a categorical variable
#       Default aggregation is by axis=0

# 3.2 Collectively we refer to the grouping
#     objects as the keys.

grouped = df.groupby('class')      # Same as: df.groupby(['class'])
grouped1 = df.groupby(['class', 'order'])

# 3.3
grouped      # <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x7f2f944e2128>
grouped1     # <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x7f2f944e27b8>


###########################
##4. GroupBy object attributes
##   TREAT GroupBy AS a Group of DATAFRAMEs
###########################

# 4.1
grouped.groups           # Describes groups dictionary
grouped1.groups          # Dict object

# 4.2
len(grouped)             # How many items are there in the group
len(grouped1)            # 6


# 4.3 Iterating through the groups
#     Peeping into each basket:

for name, group in grouped:
    print(name)
    print(group)

# 4.4 Out of these multiple boxes/groups
#     A single group can be selected using
#     get_group():

grouped.get_group('fish')


##############
## 5. Aggregating
#############
# Once the GroupBy object has been created,
# several methods are available to perform
# a computation on each one of the groups.
# PRACTICALLY CONSIDER 'grouped' AS DATAFRAME
# FOR EXAMPLE: df['max_speed'].sum()

# 5.1
grouped['max_speed'].sum()     # keys are sorted
# OR
grouped.max_speed.sum()

"""
# Summary methods to grouped objects are:

	mean() 	   Compute mean of groups
	sum() 	   Compute sum of group values
	size() 	   Compute group sizes
	count()    Compute count of group
	std() 	   Standard deviation of groups
	var() 	   Compute variance of groups
	sem() 	   Standard error of the mean of groups
	describe() Generates descriptive statistics
	first()    Compute first of group values
	last() 	   Compute last of group values
	nth() 	   Take nth value, or a subset if n is a list
	min() 	   Compute min of group values
	max()      Compute max of group values

"""

# 5.2 With grouped Series you can also pass a
#     list or dict of functions to do
#     aggregation with, outputting a DataFram

grouped['max_speed'].agg([np.sum, np.mean, np.std])

# 5.3 By passing a dict to aggregate you can apply a
#     different aggregation to the columns of a DataFrame:

grouped.agg({'max_speed': np.sum,
             'max_wt': np.std })


##############
## 6. Class Exercises:
#############

%reset -f
import pandas as pd
import numpy as np

# 6.0 Create dataframe
# 6.1 First create a dictionary
dd = {'age' :  np.random.uniform(20,30,10), 'city' : ('del', 'fbd') * 5}
dd

# 6.1 Next dataframe
abc = pd.DataFrame(dd)
abc

# 7.0 Now answer these questions
# Q 7.1. Group by city and show groups:
grouped = abc.groupby(['city'])
grouped.groups

# Q 7.2. Show minimum of age in each group
grouped['age'].min()

# Q 7.3. Just get 'del' group
grouped.get_group('del')

# 8. Change the above dataframe as follows:

dd['gender'] = list('mmmmmmfffm')
dd['income'] = np.random.random(10)
dd
cde = pd.DataFrame(dd)
cde

# Q 8.1: Group by city and gender
grouped1 = cde.groupby(['city','gender'])

# Q 8.2 Find average age by by city and gender
#       Note multiple-indexes
grouped1['age'].mean()

# Q 8.3 Transform one of the indexes as columns
grouped1['age'].mean().unstack()

# Q 8.4. Find average 'age' but 'min' income by 'city' and 'gender'
grouped1['age','income'].aggregate({'age' : 'mean' , 'income': 'min'  })
grouped1['age','income'].aggregate({'age' : 'mean' , 'income': 'min'  }).unstack()


# Q 8.5. Apply multiple functions on each numerical column:

grouped1['age','income'].aggregate({'age' : ['mean', 'max'], 'income' : np.min})

# Q 8.6. Design your own summary function
#        The functions input is a dataframe and output
#        is a pandas object or a scalar:

def wax(ds):
    return((np.sum(ds))**2 )

# Q 8.7. Does the function work?

wax(cde['income'])

# Q 8.8. Now use it on grouped1

grouped1['income', 'age'].agg({'age': [wax, np.sum]})

# Q 8.8.1 Rename columns

grouped1['income', 'age'].agg({'age': [wax, np.sum]}).rename(columns= {'wax': 'sum *2', 'sum': 'summation'})

# Q9. Using apply()
#     Function invoked by 'apply()':
#       i)  Splits dataframe
#	ii) Takes a function as argument
#      iii) Passes to that function, one by one
#	    split-dataframes
# 	iv) Dataframes/scalars returned by function
#	    are combined into a DataFrame or a Series
#
# 	Function must return a scalar or
#       a pandas object. In between what you do in
#       the function is your business

grouped1['income', 'age'].apply(wax)
grouped1['income', 'age'].apply(lambda r: np.sqrt(r))


# Q10. Using transform() function: Feature creation
grouped1['income', 'age'].transform(wax)

# Q10 Group by 'city' and summarise gender

##############################################3

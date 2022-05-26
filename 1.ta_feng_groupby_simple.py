"""
Last amended: 16th January, 2019
My folder: /home/ashok/Documents/ta_feng_grocerystore
           C:\Users\ashok\OneDrive\Documents\Ta Feng Grocery Datasets

Ref:
     https://pandas.pydata.org/pandas-docs/stable/cookbook.html#cookbook-grouping
     https://pandas.pydata.org/pandas-docs/stable/groupby.html

####################################################################3
# Ta Feng Grocery dataset
# Data Source: http://stackoverflow.com/questions/25014904/download-link-for-ta-feng-grocery-dataset
#	Other grocery datasets:
#		https://sites.google.com/a/dlpage.phi-integration.com/pentaho/mondrian/mysql-foodmart-database/foodmart_mysql.tar.gz?attredirects=0
#		http://recsyswiki.com/wiki/Grocery_shopping_datasets
# References on clustering/customer segmentation or on kohonen SOM:
# 		1. https://cran.r-project.org/web/views/Cluster.html
#   	2. http://www.shanelynn.ie/self-organising-maps-for-customer-segmentation-using-r/
#		3. General on kohonen: https://dzone.com/articles/self-organizing-maps
#		4. http://www.slideshare.net/jonsedar/customer-clustering-for-marketing
#
####################################################################

# Objectives:
            1. Reducing dataframe memory
            2. Understand customer behaviour (using pandas groupby)
            3. Chi-square Tests and mosaic plot
            4. t-test of means
            5. Feature Engineering

 ----Customer Behaviour----
 For  every customer (unique customerid)
	  Record his first purchase date
	  Record his last purchase date
	  Which customers purchase just one-time
	  Who are repeat purchasers	or who have visited more than once
	  Record every customers total purchases
	  Record every customers average purchases
	  Record his basket of purchases: Variety of goods he purchases
   Per visit/per transaction min. max items purchased and avg money spent
 For your store:
	  What is the distribution of customers age-wise?
	  What is the distribution of age, res-area-wise
	  Age wise what is the average purchase basket
   Is there age preference for a particular product sub-class
 For a product-subclass
	  Which product-subclass brings most revenue
   Which productids are most popular
	Which productIds are most costly
   And which customers purchase them?
 Tests:
   IS there a relationship between age and product_subclass
   IS there a relationship between residence_area and product_subclass
   IS there a relationship between age and residence_area

   Is there significant difference in avg spending, age-wise
   Is there significant difference in avg spending, residence wise

 Questions not answered
   Which days of week show heavy spending
   Which days of month show heavy spending
   Which days of week show least spending
   which days of month show leat spending
   What is customer life-time value
   What are quarter-wise sales of product (use thicken())
# *******************************
 ----Feature Engineering ----
# *******************************
  i)  Add a column to d12, that has counts of product_subclass
      as many times as it occurs (FE1)
  ii) Add a column to d12, that has counts of (age, residence_area)
      as many times as the combination occurs (FE2)
 iii) Add a column that has variance of (age,residence_area) wise spending (FE3)
 iv)  Create a loop for the purpose (FE4)


"""

# 1.0 Reset memory
#     ipython magic command
%reset -f
import pandas as pd
import numpy as np


# 1.1 For chi-square tests
from scipy.stats import chi2_contingency
# 1.2 For t-test
from scipy.stats import ttest_ind
# 1.3 Finding out score at a percentile point and
#     pearson correlation coeff function
from scipy.stats import scoreatpercentile, pearsonr


# 1.4
import matplotlib.pyplot as plt
import seaborn as sns
# 1.4.1 Mosaic plots
# https://www.statsmodels.org/dev/graphics.html
# https://www.statsmodels.org/dev/generated/statsmodels.graphics.mosaicplot.mosaic.html#statsmodels.graphics.mosaicplot.mosaic
from statsmodels.graphics.mosaicplot import mosaic


# 1.4 Misc facilities
from collections import Counter
import os, time, sys, gc

# 1.5 Display as many columns as possible
pd.set_option('display.max_columns', 500)


# 2.0 Set working folder and list files
os.chdir("/home/ashok/Documents/3.ta_feng_grocerystore")
#os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\Ta Feng Grocery Datasets")

os.listdir()

# 2.1 Read file at the same time reduce memory usage
#df = reduce_mem_usage(pd.read_csv("dall.csv"))
# 2.1.1 Or read directly from zip file and also save memory
df = reduce_mem_usage(pd.read_csv("dall.csv.zip"))
gc.collect()


# 2.2 Explore data
df.shape            # (817689, 8)
df.head()
df.columns
df.columns.values   # 'datetime', 'customerid', 'age', 'residence_area',
                    # 'product_subclass', 'productid', 'quantity', 'asset', 'salesprice'

df.values           # Complete dataframe as an array

# 2.3
sys.getsizeof(df)           # 201,969,287 bytes  195,427,775; 183980129
np.sum(df.memory_usage())   # 58,873,688; 40,884,530

# 2.4 Data types
df.dtypes

# 3. We have no use for 'asset' column. Drop it
df.drop(columns = ['asset'], inplace = True)

# 3.1
# To further save memory, transform following five attributes
#  to pandas 'category' type
#  customerid (int64), product_subclass (int64),
#  residence_area (object), age (object), productid (int64)
#  Remember: This is only a space saving measure.

df['customerid'] = df['customerid'].astype('category')
df['product_subclass'] = df['product_subclass'].astype('category')
df['residence_area'] = df['residence_area'].astype('category')
df['age'] = df['age'].astype('category')
df['productid'] = df['productid'].astype('category')
gc.collect()            # Release memory back to system


# 3.2 So what is our file size now?
sys.getsizeof(df)     # 76,924,417 ; 68,747,527


# 4. Conversion of datetime to 'datetime' datatype.
 #    This will further save memory.
 #    'object' is most general type of datatype
 #     Transforming to known types, saves space

df['datetime'] = pd.to_datetime(df['datetime'])
df['datetime'].dtypes.name            # datetime64[ns]


# 4.1 What is the filesize now?
sys.getsizeof(df)        # 28,680,766; 20503876
gc.collect()             # Garbage collection


# 4.2 To extract year, month, day`etc operate as follows:
# Ref:https://pandas.pydata.org/pandas-docs/version/0.22/api.html#datetimelike-properties
#     https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.html

# 4.2.1 OR, as follows (note the use of 'dt' method)
df['datetime'].dt.year
df['datetime'].dt.month
df['datetime'].dt.day

## Start asking questions

# 5 (Q1) How many unique customers, productids & product_subclasses exist
df['customerid'].nunique()                      # 32266
df['customerid'].value_counts()                 # 32266
df['customerid'].value_counts().shape           # 32266

df['productid'].nunique()                       # 23812
df['product_subclass'].nunique()                # 2012

####### Groupby:

# 5.1 (Q2) Who are the oldest customers?
#          Sol: Gr by customerid and find
#               earliest purchase date of every customer

# Show some groupby operations here

result = df.groupby('customerid')['datetime'].min().sort_values(ascending=False)
type(result)             # Pandas Series
result.head()
result.tail()
result.size              # 32266


# 5.2 (Q3) Recency: Find the last purchase date of every customer
#          Sol:Group by customerid and find the last date of purchase

result1 = df.groupby('customerid')['datetime'].max().sort_values(ascending = False)
result1.head()
result1.size    # 32266


## 5.3 (Q4): Repeat customers: Find repeat customers
##           Many ways to find repeat-customers
##           a. Group customers by both cid and date and count
##           b. Group the earlier table by cid and count
##           c. Use apply with function
##           d. Use apply with lambda

# 5.4 Method 1: In each box how many unique datetime are there
out = df.groupby('customerid')['datetime'].nunique()
out
out[out > 1].sort_values(ascending = False)

#5.5 Use apply()
# 5.5.1 First define a function:
#       Function must return either a scalar or a pandas object
#       Its argument is a dataframe

def atx(fd):
    return fd.datetime.nunique()


# 5.5.2 First group by 'customerid'
grouped = df.groupby('customerid')


# 5.5.3 Extract a data subset
#       Each subset is a dataframe
r1 = grouped.get_group(1975543)
r1                  # It is a dataframe
type(r1)
r2 = grouped.get_group(915939)
r2                 # Another dataframe

# 5.5.3 To test, apply the function
#       to each subset
atx(r1)
atx(r2)


# 5.5.4 Finally apply the function tx()
#       ti each grouped-subset
grouped.apply(atx).sort_values(ascending = False)

# 5.6 Use lambda
#     Each 'x' passed in lambda is a babay-dataframe
#     And value returned is True/False
grouped = df.groupby('customerid')

result = grouped['datetime'].apply(lambda x : x.nunique() > 1).sort_values()
result.head()



# 6.  (Q5): What is total no of visits of a customer during the period of data
#     Sol:Group by customerid, find distinct dates & count them
"""
# Which aggregating functions will work?
   The aggregating functions above will exclude NA values.
   Any function which reduces a Series to a scalar value is
   an aggregation function and will work, a trivial example
   is df.groupby('A').agg(lambda ser: 1).

"""

out = df.groupby('customerid')['datetime'].nunique().sort_values(ascending=False)
out.head()
out.tail()


# 6.1
# How are the following two different?
grouped = df.groupby('customerid', as_index = False)
grouped1 = df.groupby('customerid', as_index = True)

grouped['datetime'].nunique().head()
grouped1['datetime'].nunique().head()


# 6.2 Let us see distribution of visit-frequencies
# 6.2.1  First convert out (pandas Series) to DataFrame
out.name           # This series name will become column name
out.name = "freq"
out = pd.DataFrame(out)   # column name is now 'freq'
# 6.2.2 Draw boxplot now
out.boxplot(column = 'freq')
plt.ylim((0,20))
plt.show()


# 6.3 Let us verify the above results for at least one customerid
df.loc[df['customerid'] == 439725].nunique()

# 6.4 It can also be done in one line:
#     Note that this time we are calling plot.bar() &
#    not plt. Former is pandas function and later is matplotlib
#    function

plt.figure()
grouped['datetime'].nunique().value_counts().sort_values().plot.bar()
plt.show()

# 6.4.1 Delete not needed data
del xyz, grouped
del out
gc.collect()


# 7 (Q6): What are total spending per customer
#    Sol: Group by customerid and sum up purchases

# 7.1 First create a new column 'purchase'
df['purchase'] = df['quantity'] * df['salesprice']

# 7.2 Now calculate total purchases per customer
out = df.groupby('customerid')['purchase'].sum().sort_values(ascending = False)
out.head(20)

out = pd.DataFrame(out)
out.head()


# 7.2.1 Let us have a density plot of these purchases
out.name     # 'purchase'
out = pd.DataFrame(out)

plt.figure()
out.plot.kde()     # Pandas kernel denity plot
plt.xlim(-0.2e8, .1e8)
plt.show()


# 7.2.2 So there is an outlier, let us remove
#       it and then plot. We will cutoff at 99th percentile
#       At what value of data 99th percentile occurs?
scoreatpercentile(out.values, per = 99)
# 7.2.3 Here is remaining data
out = out.loc[out['purchase'] < 52135, :]


# 7.2.4 Plot density plot now
plt.figure()
out.plot.kde()
plt.show()

del out
gc.collect()


# 8. Is there any relationship between freq of visits and total purchases
#    We will use two methods:

## Method 1

# 8.1   First customer-wise visit frequencies
freq = df.groupby('customerid')['datetime'].nunique()
# 8.2 Then customer-wise purchases
purchases = df.groupby('customerid')['purchase'].sum()
freq.name = "freq"
# 8.3 Create a dataframe of the two continuous series
freq_purchases = pd.concat([freq,purchases], axis = 1)
freq_purchases.head()

# 8.4 So what kind of relationships exist?
#     Answer: Very weak!
freq_purchases.corr()


# 8.5 Include only points which are below 99th percentile of purchases
freq_purchases = freq_purchases[freq_purchases['purchase'] < 52135]

# 8.6 Let us revisit relationships
#     Relationship is stronger though not very strong
freq_purchases.corr()


# 8.7 Plot and also annotate now
#     First through pandas plotting methods
plt.figure()
freq_purchases.plot.scatter(x = 'freq', y = 'purchase')
plt.show()

# 8.7.1 Then through seaborn
plt.figure()
g = sns.jointplot("freq", "purchase", freq_purchases, kind = "reg")
g = g.annotate(pearsonr)        # Correlation of 0.51 is quite low
plt.show()


## Method 2
# 8.8 Use of aggregate() method
result = df.groupby('customerid').agg({'datetime' : 'nunique' , 'purchase' : 'sum'})
result.head()


# 9. (Q7):  What are average purchases per-customer, per visit
#     Sol: Simple: Gr by customerid,datetime and sum up purchases
#
result = df.groupby(['customerid', 'datetime'])['purchase'].sum()
result.head()

# 9.1 Note that grouping by more than one attribute
#     creates multiindex. Here it has two levels

result.index.get_level_values(0)
result.index.get_level_values(1)

# 9.2 Unentangle it using unstack()
result1 = df.groupby(['customerid', 'datetime'])['purchase'].sum().unstack()
result1.head()

# 9.3 Find mean, across columns
#     And this our answer to the question
df.groupby(['customerid', 'datetime'])['purchase'].sum().unstack().mean(axis = 1, skipna = True).head()


# 9.4 Let us verify the result for at least one customer
df[df['customerid'] == 1069].groupby('datetime')['purchase'].sum()
(187 + 971 + 922 + 580)/4               # 665


# 10 (Q8):  What are average purchases customer-wise
#          Sol: Gr by customerid and find mean() purchases per id

df.groupby('customerid')['purchase'].mean().sort_values(ascending = False)

## 11. Q9
# 11.1 Determine customer-wise, product_subclass preference?
#     We will define preference as no of different datetimes purchased.
#     All purchases of a product_subclass on one date count to one.

result = df.groupby(['customerid', 'product_subclass'])['datetime'].nunique().sort_values(ascending= False)
result.head()

# 11.2 Just to extract result for one customer from this multiindex, use
#      index.get_level_values()....rather complicated but works
#  Ref: https://stackoverflow.com/questions/17921010/how-to-query-multiindex-index-columns-values-in-pandas
result.loc[result.index.get_level_values('customerid')  == 1740653, :]


# 11.3 But if preference means by quantity, then solution is:
df.groupby(['customerid', 'product_subclass'])['quantity'].sum().sort_values(ascending= False)


# 12 (Q10): Basket of purchases:
#            Variety of purchases made per customer, Productid wise

df.groupby(['customerid'])['productid'].nunique().sort_values(ascending = False)

# 13 (Q11): Which product_class brings most revenue
#           Sol: Group by product_subclass and add quantity * salesprice

# 13.1 Here is the solution:
df.groupby(['product_subclass'])['purchase'].sum()


# 13.2 (Q12): Which product_subclass is popular that is
#             most customers buy?
df.groupby(['product_subclass'])['customerid'].nunique().sort_values(ascending=False)

## Not solved. Students to solve these
# 13.3 (Q13): Age wise purchases average. Which age group max purchases
#             Just group by age

# 13.4 (Q14): Residence area wise purchasing capacity
#             Just group by residence area wise


# 13.5 (Q15)  Per visit/per transaction avg money spent


# 14   (Q16): Distribuiton of age groups with residence_area
#             Same as:  table(d12$age, d12$residence_area)

# 14.1 First look at the help
?pd.crosstab

# 14.2 Next create contingency table
pd.crosstab(df['age'],df['residence_area'], normalize = False)      # All values are > 5

# 14.3 Which one of the following two gives a better picture?
pd.crosstab(df['age'],df['residence_area'], normalize = 'index')    # OR 'all', 'columns'
pd.crosstab(df['age'],df['residence_area'], normalize = 'index')    # OR 'all', 'columns'

# 14.4 Let us plot a barchart
ct = pd.crosstab(df['age'],df['residence_area'], normalize = 'index')
type(ct)         # Pandas DataFrame

# 14.5
ct.index         # age
ct.columns       # residence_area

# 14.3 Plot now vertically and then horizontally
ct.plot.bar()
#ct.plot.barh(figsize=(20,20))       # Increase fig size to bring clarity


# 15.(Q17)    What is the distribution of customers, age-wise
#             Not solved


# 16. (Q18) Is there a relationship between 'age' and 'residence_area'
#           We will calculate ch-square statistics
#           An often quoted guideline for the validity of chi-square calculation
#           is that the test should be used only if the observed and expected
#           frequencies in each cell are at least 5.

# 16.1 Return values are: chi2 statistic, p-value, degrees-of-freedom, expected-freq
chi2_contingency(pd.crosstab(df.age, df.residence_area))     # p-value = 0
# OR
# 16.2
chi2, p_value, dof, expeFreq = chi2_contingency(pd.crosstab(df.age, df.residence_area))
p_value



# 17. Mosiac plot of contingency table
#     At a glance view of deviation from expected freq
# statistic=True  will give colors to the plot. If the tile has a freq
#   is more than 2 standard deviation from the expected value
#    color will go from green to red (for positive deviations, blue otherwise)
#     and will acquire an hatching when crosses the 3 sigma.
fig = plt.figure(figsize = (10,10))     # Set figure size
ax = fig.add_subplot(111)               # Add one subplot
mosaic(df, ['age', 'residence_area'],
       ax = ax,
       statistic = True
       )
plt.show()


# 18   t-test IS there any significant difference in avg spending
#        age-wise, say, between ages 'A' and 'B'
#   Steps:
#        1. Extract spending data for two ages
#        2. Discover 99th percentile points for each
#        3. Remove outliers
#        4. Perform t-test

# 18.1 'purchase' data for age = 'A'
a = df.loc[df['age'] == 'A', 'purchase']
len(a)              # 30068

# 18.2 'purchase' data for age = 'B'
b = df.loc[df['age'] == 'B', 'purchase']
len(b)              # 66427

# 18.3 Conduct t-test now. Are the means of two purchases equal?
_, pvalue = ttest_ind( a, b, axis=0)
pvalue      # 0.18781 So no difference in purchase mean

# 18.4 Let us remove few outliers and then again perform t-test
#      What is the 99th percentile in each case
scoreatpercentile(a, per = 99)             # 2558
scoreatpercentile(b, per = 99)             # 3029

# 18.5 Extract from 'a' & 'b' values other than outlers
a= a[a<2558]
b = b[b<3029]
len(a)        # 29767
len(b)        # 65762

# 18.6 Perform t-test now:
# 18.7 Removing outliers drastically alters the conclusion
_, pvalue = ttest_ind( a, b, axis=0)
pvalue             # 3.060e-42


#################### I am done ##########################################

# Reduce memory
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# -*- coding: utf-8 -*-
"""
Last amended: 13/05/2018
My folder: 
    C:\Users\ashok\OneDrive\Documents\talkingdata
Data folder:
    f:\talkingdata
Accompanying file: 
    td.py

Objective:
    i)   Using dictionary to group by
    ii)  Ist case: Writing our own summary function for groupby op
    iii) IInd case: Using pandas/numpy summary functions
    iv)  IIIrd case: Specifying lambda functon to groupby in a dictionary


Summary of Feature Engineering in file td.py:
    We will be grouping. Grouping is the process
    of split-apply-combine. We can consider the process, as:
    First, split the data based upon some groups of attributes,
    decide to summarise on some other attribute and, select
    the summary function. SQL makes it very clear:
        
        SELECT Col1, Col2, mean(Col3), sum(Col4), lambdaFunc(Col5)
        FROM SomeTable
        GROUP BY Col1, Col2

    We will put these in dictionary form, as:
        { 'groupby' : ['Col1', 'Col2'] ,
          'applySummaryFuncOn' : ['Col3'] ,
           'agg' : ['mean']  }

    We will be doing grouping in two stages. In the first case, our sole attribute
    for summarising will be 'is_attributed', the target itself.
    In the IInd case, we will be selecting attributes for summary other than
    'is_attributed'.
    
    Summary function(s):
        First case     : rate_calculation()
        Second case    : var, count, cumcount, mean, nunique, lambda  

"""


# 1.0 Import libraries
%reset -f
import gc                     # Garbage collection
import numpy as np            # n-dimensional array manipulation
import pandas as pd           # Data frame related operations


####################################################################
# Part I--First case    Customised summary function
####################################################################

# 2. Define a function conf_rate
#    Define our aggregation function
# 2.1   Normalizing value
log_group = np.log(100000) 

# 2.2
# x is a boolean value. We are interested whenever x is '1'
#   As to explanation of rate_calculation() see file td.py
def rate_calculation(x):
    """Calculate the attributed rate. Scale by confidence"""
    rate = x.sum() / float(x.count())
    conf = np.min([1, np.log(x.count()) / log_group])
    return rate * conf

# 2.3 Our sample data frame
df= pd.DataFrame(
                 {'ip'  :           [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                  'app' :           [4, 4, 5, 5, 5, 4, 4, 5, 4, 4, 5],
                  'is_attributed' : [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
                  }
                )



df  

# 3. This is what we intend to do
spec = { 'groupby' : ['ip','app'] ,
          'applySummaryFuncOn' : ['is_attributed'] ,
           'agg' : ['rate_calculation']  }



# 3.1 We name the created feature as:
#     '_'.join(cols): Joins elements of list, separated
#                     by preceding symbol, ie '_'
new_feature = '_'.join(spec['groupby'])+'_confRate'  
new_feature


# 3.2 Create first a groupObject   
#     Ref: https://pandas.pydata.org/pandas-docs/stable/groupby.html 
group_object= df.groupby(spec['groupby'])    

# 2.4 Some experiments on groupObject
group_object.size()
group_object.first()
group_object.last()

# 5 Apply groupby object
group_object['is_attributed'].mean()    # Calculate mean of column 'c'
group_object[spec['applySummaryFuncOn']].mean()    # Calculate mean of column 'c'


# 5.1 Following does not work
result = group_object[spec['applySummaryFuncOn']].apply(spec['agg'])

# 5.2 Following does
group_object[spec['applySummaryFuncOn']].mean()
# 5.3 This does not work
group_object[spec['applySummaryFuncOn']].rate_calculation()
# 5.4 Do it as follows:
group_object[spec['applySummaryFuncOn']].apply(rate_calculation)



# Merging operation
#      i) Reset index
#     ii) Rename column
#    iii) Finally merge

# 6.1 Pre-merging. Reset index
result = group_object[spec['applySummaryFuncOn']].apply(rate_calculation)
dt= result.reset_index()
type(dt)
dt.columns     
dt                 # Note that last column name is still 'c'

# 6.2
# Column renaming. Default name of summmary column is same as
#  the group_on column
dt.index         
d_final=dt.rename( 
                 index=str,
                 columns={'is_attributed': new_feature}
                )
d_final


# 6.3 Or all above in one go

df.groupby(spec['groupby'])[spec['applySummaryFuncOn']]. \
       apply(rate_calculation).reset_index(). \
       rename(
               index = str,           ## ????? CLARIFY WHAT IS str
               columns = {'is_attributed' : new_feature}
               )




# 6.4 Finally merge df with xx. Put 'df' on left
df.merge(d_final, on = spec['groupby'], how = 'left')


####################################################################
# Part II--Second case     numpy summary functions
####################################################################

# 7
df= pd.DataFrame(
                 {'ip'  :           [1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 3 , 3 ,  3 ],
                  'app' :           [4 , 4 , 5 , 5 , 5 , 4 , 4 , 5 , 4 , 4 ,  5 ],
                  'os'  :           [30,18 ,30 ,18 ,18 ,19 ,19 ,30 , 30,18 , 18 ],
                  'day' :           [1 , 0 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 ,  0 ]
                  }
                )


# 7.1
df 
df.shape              # (11,4)


# 7.2
spec = { 'groupby' : ['ip','app'] ,
          'applySummaryFuncOn' : ['day'] ,
           'agg' : ['mean']  }


# 7.3 Name the new featureWe name the created feature as:
new_feature = '_'.join(spec['groupby'])+"_"+spec['agg'][0]  
new_feature

#7.4
dx = df.groupby(spec['groupby'])
dx
#7.5
dx.mean()[spec['applySummaryFuncOn']]
# 7.6
dx = dx.agg(spec['agg'])

# 8. All in one chain: extract just the column you want
dt = df.groupby(spec['groupby']).agg(spec['agg'])[spec['applySummaryFuncOn']]
dt.columns
dt.index 


# Merge df and dt
# 8.1 Pre-merging
dt= result.reset_index()
type(dt)
dt.columns     
dt

# 8.2 Renaming column        
d_final=dt.rename( 
                 index=str,
                 columns={'is_attributed': new_feature}
                )
d_final



# 6.3 Or all above in one go
######### ???????
df.groupby(spec['groupby'])[spec['applySummaryFuncOn']]. \
       apply(rate_calculation).reset_index(). \
       rename(
               index = str,           ## ????? CLARIFY WHAT IS str
               columns = {'is_attributed' : new_feature}
               )





# 8.3 Now merge
# 9 Merge df with xx. Put 'df' on left
df.merge(d_final, on = spec['groupby'], how = 'left')


####################################################################
# Part III--Third case    lambda
####################################################################


# 10
df= pd.DataFrame(
                 {'ip'  :           [1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 3 , 3 ,  3 ],
                  'app' :           [4 , 4 , 5 , 5 , 5 , 4 , 4 , 5 , 4 , 4 ,  5 ],
                  'os'  :           [30,18 ,30 ,18 ,18 ,19 ,19 ,30 , 30,18 , 18 ],
                  'day' :           [1 , 0 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 ,  0 ]
                  }
                )


# 10.1
df 
df.shape              # (11,4)

# 11. Dictionary of operations
spec = {'groupby': ['app'], 'applySummaryFuncOn': 'ip', 
                          'agg': lambda x: float(len(x)) / len(x.unique()), 
                          'agg_name': 'AvgViewPerDistinct'
                          }


# 11.1 Name the new featureWe name the created feature as:
new_feature = '_'.join(spec['groupby'])+"_"+spec['agg_name']
new_feature

# 11.2 All in one
xp = df.groupby(spec['groupby'])[spec['applySummaryFuncOn']].agg(spec['agg'])
xp=  xp.reset_index().rename(index=str, columns={spec['applySummaryFuncOn']: new_feature})
xp       


# 11.3 Or rather:
#      Pick only features of interest
#      Why carry extra-lugg
#      Make it work faster
all_features = ['ip','app']

# 11.4
d_final = df[all_features].groupby(spec['groupby'])[spec['applySummaryFuncOn']]. \
     agg(spec['agg']). \
     reset_index(). \
     rename(index=str, columns={spec['applySummaryFuncOn']: new_feature})
d_final
 
# 11.5 Chain all operations together
d_final = df[all_features]. \
       groupby(spec['groupby'])[spec['applySummaryFuncOn']]. \
       agg(spec['agg']). \
       reset_index(). \
       rename(index=str, columns={spec['applySummaryFuncOn']: new_feature})
        
d_final

# 11.6 Now merge
# 9 Merge df with xx. Put 'df' on left
df.merge(d_final, on = spec['groupby'], how = 'left')


#################################################################              
## TODO
################################################################3


#****** Case 3: cumcount ********

df= pd.DataFrame(
                 {'ip'  :           [1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 3 , 3 ,  3 ],
                  'app' :           [4 , 4 , 5 , 5 , 5 , 4 , 4 , 5 , 4 , 4 ,  5 ],
                  'os'  :           [30,18 ,30 ,18 ,18 ,19 ,19 ,30 , 30,18 , 18 ],
                  'day' :           [1 , 0 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 ,  0 ]
                  }
                )



df 
df.shape              # (11,4)

spec = {'groupby': ['ip'],  'applySummaryFuncOn': 'app', 'agg': 'cumcount'}

agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
print(agg_name)
# Name of new feature
new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['applySummaryFuncOn'])
print(new_feature)
# Info
print("Grouping by {}, and aggregating {} with {}".format(
      spec['groupby'], spec['applySummaryFuncOn'], agg_name ))

# Unique list of features to select. Save memory.
all_features = list(set(spec['groupby'] + [spec['applySummaryFuncOn']]))
print(all_features)
# Perform the groupby
xp = df[all_features]. \
       groupby(spec['groupby'])[spec['applySummaryFuncOn']]. \
       agg(spec['agg']). \
       reset_index(). \
       rename(index=str, columns={spec['applySummaryFuncOn']: new_feature})
        
print (xp)     
        
# Merge back to df
if 'cumcount' == spec['agg']:
    df[new_feature] = xp[0].values
else:
    df = df.merge(xp, on=spec['groupby'], how='left')
    
print(df)    


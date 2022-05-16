# Last amended: 2nd March, 2021
# Myfolder: C:\Users\Administrator\OneDrive\Documents\python\basic_lessons
# Accomanying Excel file: groupingData.xlsx

import pandas as pd
import numpy as np

# 1.0
df = pd.DataFrame([
             ('AA', 'XX', 'less', 45.7,12.5,41),
             ('AB', 'XY', 'more', 45.7,12.7,43),
             ('AA', 'XX', 'less', 45.7,12.8,44),
             ('AA', 'XY', 'more', 45.9,12.9,44),
             ('AB', 'XX', 'more', 44.8,12.9,44),
             ('AA', 'XY', 'more', 43.1,12.9,43),
             ('AB', 'XX', 'less', 46,15.1,44.2),
             ],
                  columns = ['X1', 'X2','X3','X4','X5','X6']
                 )

#art = np.array(['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a3'])
#art.shape


# 1.1
df
df.shape

# 2.0 Summarise whole of data:
df.mean()
df.std()
df.median()
df.mode()

# 2.1 Better clarify summarisation
# 2.1.1 One way:
df[['X4','X5','X6']].mean()
df[['X1','X2']].mode()

# 2.1.2 Another way
df.agg({'X4': 'mean'})
df.agg({'X4': np.mean})
df.agg({'X4' : np.mean, 'X5': np.median})

##############
##############

# 3.0 Group operation
grpd = df.groupby([df['X1'],df['X2']])
grpd = df.groupby(['X1', 'X2'])

# 3.1
grpd.mean()
grpd.median()
grpd.size()

# 3.2 Better clarify
grpd['X4'].mean()
grpd[['X4','X5']].mean()
grpd.agg({'X4' : [np.mean, np.median] , 'X5' : np.sum})
grpd.apply(np.mean)

#####################
# Dealing with NAs
####################

# 4.0
df = pd.DataFrame([
             ('AA', 'XX', 'less', 40,50,50),
             ('AA', 'XX', 'more', np.nan,20,40),
             ('AA', 'XX', 'less', 10,60,np.nan),
             ('AB', 'XY', 'more', np.nan,30,40),
             ('AB', 'XY', 'less', 80,90,np.nan),
             ('AB', 'XX', 'more', 20,20,40),
             ('AB', 'XX', 'less', 50,np.nan,30),
             ('AA', 'XY', 'more', 40,10,40),
             ('AA', 'XY', 'more', 43.1,12.9,43),
             ],
                  columns = ['X1', 'X2','X3','X4','X5','X6']
                 )
# 4.1
df


# 4.2 Calculate overall mean/median per column:
t = df['X4'].mean()
t

# 4.3 Fill with data mean/median
df['X4'].fillna(value = t)

# 5.0 Fill with group mean
#     Use apply()

#5.1
grpd = df.groupby(['X1','X2'])

# 5.2
def avg(x):
    t = x.mean()
    x = x.fillna(value = t)
    return x

# 5.3
grpd = df.groupby(['X1','X2'])
grpd['X4'].apply(avg)



## 6.0 Feature creation:

# 6.1
grpd.size()

# 6.2
r = grpd.size()
r

# 6.3
r.index
# 6.4
r.shape
# 6.5
type(r)

# 7.0 Two ways to simplify
#      multiple index
# 7.1
r.unstack()
# 7.2
r.reset_index()

# 7.3 Merge two dataframes on common columns
t = r.reset_index()
df.merge(t)

# 7.4 Rename column
df.merge(t).rename({0 : 'cnt' }, axis =1)

# 8.0
# Distribution of categorical features
# Contingency table:

pd.crosstab(df['X1'], df['X2'])


#######################################

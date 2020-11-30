# Last amended: 30th Nov, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\useful_code & utilities
# Ref: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
#      https://stackoverflow.com/a/46498792/3282777
# Objective:
#           Discover association between nominal features
#           Varies between [0,1]. 0=>No association
#           1=>Complete association

# 1.0 Call libraries
import pandas as pd
import numpy as np
import scipy.stats as ss

# 1.1 cramers function
def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# 1.2 Usage:
import seaborn as sns
tips = sns.load_dataset("tips")
tips["total_bill_cut"] = pd.cut(tips["total_bill"],
                                np.arange(0, 55, 5),
                                include_lowest=True,
                                right=False)

confusion_matrix = pd.crosstab(tips["day"], tips["time"]).values
cramers_v(confusion_matrix)

confusion_matrix = pd.crosstab(tips["total_bill_cut"], tips["time"]).values
cramers_v(confusion_matrix)
##################

---
title: Application Scorecard using Deep Learning
date: 2019-12-29 21:58:47
tags: [Credit Risk, Application Socrecard, Deep Learning]
---

## Develop a Risk Scorecard using Deep Learning and other ML algorithms

In the below article I will be taking you through the end to end process of developing a Credit Risk Scorecards. These scorecards are typically used by the Underwriting team of a Bank who takes this score to access the Risk attached to a person. By Risk, what I mean is what is the probability that the person can default on the loan given.

Let's start the process!

### Importing the libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
import random
from scipy import stats

warnings.filterwarnings('ignore')
```

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
```

### Advanced Classifiers
```python
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
```

### Oversampling of the Data
```python
from imblearn.over_sampling import SMOTE
```

### Deep learning framework
Firstly you need to install tensorflow to be able to use keras, the deep learning framework. Also make sure that tensorflow version is above 2.2 else you won't be able to use keras. So, things to do before running the below codes:

* pip install tensorflow 
* import tensorflow as tf
* pip install keras 

```python
from keras import models
from keras import layers
```

### Reading the Data
```python
data_  = pd.read_csv('D:/External Data/GiveMeSomeCredit/cs-training.csv')
```
### Check the percentage of Defaults and Non-Defaults
It is important to check the Default Rates in case you are creating a scorecard. What is does help in understanding about the data that whether we have a imbalanced data. An imbalanced data is basically where the event rate is really low; in the range of 0-5%. Usually, the defaults are a rare scenarios in case of Home Loans & Auto Loans. When you have a default rate in this range, we should be able to balance the data first and then run any model on the same. To balance a data set, there are a different appraoches but the most common and the one which we will use are:

* SMOTE (Synthetic Minority Over-sampling Technique)
* Near Miss (Under sampling Technique)

Below image clearly explains How Undersampling and Oversampling works:
![Undersampling and Oversampling](https://oralytics.files.wordpress.com/2019/05/screenshot-2019-05-20-15.34.14.png?w=705)
Image Source: https://oralytics.com/2019/07/01/managing-imbalanced-data-sets-with-smote-in-python/

**SMOTE** works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line. Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space.

**NEAR MISS** is an under-sampling technique. It aims to balance class distribution by randomly eliminating majority class examples. When instances of two different classes are very close to each other, we remove the instances of the majority class to increase the spaces between the two classes. This helps in the classification process. To prevent problem of information loss in most under-sampling techniques, near-neighbor methods are widely used.

### Checking for the Default rate
As we see that it is definitely a case of imbalanced data. So, we had to follow one of the approach discussed above; so either we will be oversampling or undersampling. There is a trade-off between having to chooses undersampling or oversampling:

1. **With undersampling**, there is a big loss of data as the total number of rows will be reduced to the number of rows for event rate (Defaults). So, there is a high possibility of High Bias,
1. **With oversampling**, there will be synthetic data added to make the Defaults observations equal to the Non-Defaults. The synthetic data is not the exact replica of the observations under Default but these are very much the same as chosen from their nearest neighbours. This can really increase the variance in our model which leads to overfitting. The other disadvantage of oversmapling is that it can increasre the model run time if we are dealing with large number of Non-Defaults

```python
print(data_['SeriousDlqin2yrs'].value_counts()/data_.shape[0] *100)

## Pie Chart
labels = 'Default', 'Non-Defaults'
sizes = [6.684, 93.316]
explode = (0.2, 0)
cols    = ['#00FFFF', '#008080']

fig = plt.figure(figsize = (4,4))
plt.pie(sizes, explode=explode, colors = cols, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt. title("Percentage of Defaults and Non-Defaults")
plt.show()
```

## Figure 1

### Exploratory Data Analysis (EDA)
These will be the steps we will follow before trying to develop any model out of the model:

* Identification of Missing values
* Treatment of Missing values
* Univariate Analysis for all the variables
* Bivariate Analysis
* Correlations

These will help us to strengthen our understanding on the data. We will be knowing about which variables have missing values, outliers if any, the general distribution. Also, we would know about the correlations between the independent and dependent variables.

#### Missing value function

```python
def missing_vals(data_):
    miss_     = data_.isnull().sum()
    miss_pct  = data_.isnull().sum()/data_.shape[0]
    
    miss_pct  = pd.concat([miss_, miss_pct], axis =1)
    miss_pct.reset_index(inplace=True)
    miss_cols = miss_pct.rename(columns={'index':'Column Name', 0:'Missings', 1:'Missing_pct'})
    
    miss_cols = miss_cols[miss_cols.iloc[:,1]!=0].sort_values('Missing_pct', ascending=False).round(1)
    miss_cols.reset_index(inplace=True, drop=True)
    
    return miss_cols    
```

```python
miss = missing_vals(data_)
miss
```

## Fig2

**Conclusion**: We see that there are very less missing values in two of the columns. Then also, we will be replacing the missing values with mean. Missing value treatment is another area which has so many options but the quickest is replacing it by mean or median (when the missing percentage is really less)

#### Descriptive Statistics
We start with the Descriptive statistics by checking the basic statistics of the variables. Then we start developing histograms for all the variables. These will help us understand the distributions of each of the variable.

```python
data_.describe()
```

## Fig3

```python
cols = list(data_.columns)
cols = cols[1:]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']

fig = plt.figure(figsize=(15, 12))
for i in range(0, len(cols)):
    plt.subplot(5, 4, i+1)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)

    plt.hist(data_[cols[i]], bins=30, color=colors[i])
    plt.title(cols[i])

plt.tight_layout()
```
#### Replace the missings with the mean

```python
df = data_.copy()
df['MonthlyIncome'] =df['MonthlyIncome'].transform(lambda x: x.fillna(x.mean()))
df['NumberOfDependents'] =df['NumberOfDependents'].transform(lambda x: x.fillna(x.mean()))
```
#### Check for missing values again

```python
miss = missing_vals(df)
miss
```

As we see that there are no null values left. Many of the Machine Learning algorithms do take missing values in their analysis but it is always a better strategy to treat them before reaching the modelling stage.

There are certain instances when you can't treat them:
1. Missing values take away some 30-50% of observations where you can't delete them
1. In the above case, you can't even treat them with mean, median as a variables with so many similar values doesn't bring a lot of variance in the variable

### Default and Non-Default Visualization
We will try to visualize the Default and Non-Defaults on a 2axis framework and see how much of the overlap they have. For this we will use PCA to get two principal components which are a combination of all the variables. This will help us understand the distribution of Defaults and Non-Defaults

```python
random.seed(32)
pca = PCA(n_components = 2)
pca.fit(df)

scores = pca.transform(df)

x,y = scores[:,0] , scores[:,1]
df_ = pd.DataFrame({'x': x, 'y':y, 'clusters':df['SeriousDlqin2yrs']})
grouping_ = df_.groupby('clusters')


fig, ax = plt.subplots(figsize=(10, 5))
names = {0: 'Non-Defaults', 
         1: 'Defaults'}

for name, grp in grouping_:
    ax.plot(grp.x, grp.y, marker='o', label = names[name], linestyle='')
    ax.set_aspect('auto')
    ax.set_ylim([0,200000])     ### I have just kept a upper cap on the axis to see the distribution of them
    
ax.legend()
plt.title('Plot showing Defaults and Non-Defaults')
plt.show()
```

### Now doing the one-hot encoding
One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction

Some algorithms can work with categorical data directly. For example, a decision tree can be learned directly from categorical data with no data transform required (this depends on the specific implementation). Many machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric. In general, this is mostly a constraint of the efficient implementation of machine learning algorithms rather than hard limitations on the algorithms themselves. This means that categorical data must be converted to a numerical form.

Below image clearly explains How one-hot encoding works:
![Undersampling and Oversampling](https://naadispeaks.files.wordpress.com/2018/04/mtimfxh.png?w=371&h=146)

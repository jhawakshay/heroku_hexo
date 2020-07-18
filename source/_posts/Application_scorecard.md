---
title: Application Scorecard using Deep Learning
date: 2019-12-29 21:58:47
tags: [Credit Risk, Application Socrecard, Deep Learning]
---

## Develop a Risk Scorecard using Deep Learning and other ML algorithms

In the below article I will be taking through the end to end process of developing a Credit Risk Scorecards. These scorecards are typically used by the Underwriting team of a Bank who takes this score to access the Risk of a person. By Risk, what I mean is what is the probability that the person can default on the loan given.

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

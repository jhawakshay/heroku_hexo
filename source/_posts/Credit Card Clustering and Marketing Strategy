---
title: k-means Clustering of Credit Card Customers
date: 2019-04-29 21:58:47
tags: [k-means, Clustering, Credit Card Marketing, Customer Behaviour]
---

## How can can we leverage Clustering to develop strategies for our Credit Card Customers?

Credit Card is a big business for any Bank. Issuing a Credit card not only a Brand awareness strategy but a revenue generating exercise too.
Credit card gives users an extra income to meet their needs which they can repay by next month to the bank without paying an interest.

The tricky part is not everyone thinks like that! Not everyone has the ability to pay his/her dues. Not everyone thinks from a Budget perspective.
Not everyone clear dues on time.

All these n number of combinations is where Banks can make money by getting the interest or what we say as a revolving customer. A revolving customer
is one who just pays the minimum amount as his/her due so that he is not considered to be missed his Credit Card Payment. 
So, it is a big opportunity for a Bank to make money but it has to work closely with its Risk as well as Strategy team which finally works 
with the Marketing team.

Enough talking and giving a lot of gyaan! Let's work on a dataset and find clusters/groups on the basis of Credit Card behaviour of customers

This dataset could be found here [Credit Card Marketing](https://www.kaggle.com/arjunbhasin2013/ccdata)

Let's start coding then

Importing python libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
```

```python
data_   = pd.read_csv('E:/Machine Learning/Kaggle/14701_19663_bundle_archive/CC GENERAL.csv')
data_.shape
data_.describe()
```

Check for missing values
```python
def missings_(data):
    miss      = data.isnull().sum()
    miss_pct  = 100 * data.isnull().sum()/len(data)
    
    miss_pct      = pd.concat([miss,miss_pct], axis=1)
    missings_cols = miss_pct.rename(columns = {0:'Missings', 1: 'Missing pct'})
    missings_cols = missings_cols[missings_cols.iloc[:,1]!=0].sort_values('Missing pct', ascending = False).round(1)
    
    return missings_cols 
    
missings = missings_(data_)
missings
```

	              Missings	Missing pct
MINIMUM_PAYMENTS	313	        3.5
CREDIT_LIMIT	     1	        0.0

Data Exploration starts!

```python
fig, ax = plt.subplots(1,4,figsize =(20,4))
ax0, ax1, ax2, ax3 = ax.flatten()

ax0.hist(data_['INSTALLMENTS_PURCHASES'], bins = 60, alpha =0.8 )
ax1.hist(data_['MINIMUM_PAYMENTS'], bins = 10, color="green" ,alpha =0.8 )
ax2.hist(data_['CREDIT_LIMIT'], bins = 60, color="red",alpha =0.8 )
ax3.hist(data_['CASH_ADVANCE'], bins = 60, color="orange",alpha =0.8 )

ax0.set_title("INSTALLMENTS_PURCHASES")
ax1.set_title("MINIMUM_PAYMENTS")
ax2.set_title("CREDIT_LIMIT")
ax3.set_title("CASH_ADVANCE")

plt.show()
```
![Histograms for variables](/images/clustering/hist1.png)

```python
fig, ax = plt.subplots(1,4,figsize =(20,4))
ax0, ax1, ax2, ax3 = ax.flatten()

ax0.hist(data_['INSTALLMENTS_PURCHASES'], bins = 60, alpha =0.8 )
ax1.hist(data_['MINIMUM_PAYMENTS'], bins = 10, color="green" ,alpha =0.8 )
ax2.hist(data_['CREDIT_LIMIT'], bins = 60, color="red",alpha =0.8 )
ax3.hist(data_['CASH_ADVANCE'], bins = 60, color="orange",alpha =0.8 )

ax0.set_title("INSTALLMENTS_PURCHASES")
ax1.set_title("MINIMUM_PAYMENTS")
ax2.set_title("CREDIT_LIMIT")
ax3.set_title("CASH_ADVANCE")

plt.show()
```
![Histograms for variables](/images/clustering/hist2.png)

```python
cols   = data_.columns
fig , ax = plt.subplots(1,4, figsize = (20,8))
ax0, ax1, ax2, ax3 = ax.flatten() 

for i in range(0,4):
    
    X   = data_[cols[i+2]]
    Y   = data_[cols[1]]
    ax[i].plot(X, Y, marker = 'o', linestyle = "None")
    ax[i].set_xlabel(cols[i+2])
    ax[0].set_ylabel(cols[1])
```
![Histograms for variables](/images/clustering/img3.png)

Some Violin plots

```python
fig = plt.figure(figsize = (50,20))
data_sub = data_[(data_['BALANCE_FREQUENCY']>=0.3)]
data_sub['BALANCE_FREQ'] = round(data_['BALANCE_FREQUENCY'],2)

sns.violinplot(y='BALANCE',x='BALANCE_FREQ',data=data_sub)
plt.xlabel('BALANCE FREQ',fontsize=40)
plt.ylabel('BALANCE',fontsize=40)
plt.tick_params(labelsize=30)
```
![Histograms for variables](/images/clustering/img4.png)

Now exploring the PURCHASES, BALANCES, ONE-OFF PURCHASES

```python
data_['Balance_decile'] = pd.qcut(data_['BALANCE'], q=10)
data_grp   = data_.groupby('Balance_decile', as_index=False).mean()
data_grp   = data_grp[['Balance_decile', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES']]
data_grp_t = pd.melt(data_grp, id_vars = 'Balance_decile')

fig = plt.figure(figsize=(20,10))
sns.barplot(x= "Balance_decile" , y = "value", hue = 'variable', data =data_grp_t)
plt.ylabel("Average Purchase Amount", fontsize=20)
plt.xlabel(" Balance Groups", fontsize =20)
plt.tick_params(labelsize=12)
plt.xticks(rotation=45)
plt.show()
```
![Histograms for variables](/images/clustering/img5.png)

Finding Average Balances by Balance Frequency
```python
data_['freq_purchase_decile'] = pd.qcut(data_['PURCHASES_FREQUENCY'], q=4)

data_bal   = data_.groupby('freq_purchase_decile', as_index=False).mean()
fig = plt.figure(figsize=(10,5))
sns.barplot(x= "freq_purchase_decile" , y = "BALANCE", data =data_bal)
plt.show()
```
![Histograms for variables](/images/clustering/img6.png)

Finding Credit Card Utilization
```python
data_['CREDIT_LIMIT'].fillna(1, inplace=True)
data_['CC_utilisation']     = (data_['CREDIT_LIMIT'] - data_['BALANCE'])/data_['CREDIT_LIMIT']

data_['CC_util_decile']     = pd.qcut(data_['CC_utilisation'], q=10)
data_cc_grp                 = data_.groupby('CC_util_decile', as_index=False).mean()
data_cc_grp                 = data_cc_grp[['CC_util_decile', 'PAYMENTS' , 'MINIMUM_PAYMENTS']]
data_cc_grp_t               = pd.melt(data_cc_grp, id_vars = 'CC_util_decile')

fig = plt.figure(figsize=(10,5))
sns.barplot(x= "CC_util_decile" , y = "value", hue = "variable" ,data =data_cc_grp_t)
plt.xlabel("Credit Card Utilization")
plt.ylabel("Average Payments")
plt.xticks(rotation=45)
plt.show()
```
![Histograms for variables](/images/clustering/img7.png)

After analyzing the data, we will create clusters for marketing strategies
Since, there are also outliers, we will be creating bins to nullify the effect of Outliers
```python
data_n  = data_.copy()

cols = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES','CASH_ADVANCE',
         'CREDIT_LIMIT', 'PAYMENTS']
for c in cols:
    bins = c+'_bin'
    max_ = max(data_n[c])
    data_n[bins] = pd.cut(data_n[c], bins=[0,500,1000,3000,5000,10000,15000,max_],labels = [1,2,3,4,5,6,7], include_lowest= True)
```

```python
cols = ['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',
         'CASH_ADVANCE_FREQUENCY']
for c in cols:
    bins = c+'_bin'
    max_ = max(data_[c])
    data_n[bins] = pd.cut(data_n[c], bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,max_],labels = [1,2,3,4,5,6,7,8,9,10], include_lowest= True)
```

```python
cols = ['CASH_ADVANCE_TRX', 'PURCHASES_TRX']

for c in cols:
    bins = c+'_bin'
    max_ = max(data_[c])
    data_n[bins] = pd.cut(data_n[c], bins=[0,20,40,60,80,100,max_],labels = [1,2,3,4,5,6], include_lowest= True)
```
Dropping some of the columns

```python
data_model  = data_n.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'Balance_decile',
       'freq_purchase_decile', 'CC_utilisation', 'TENURE', 'PURCHASES_TRX_bin', 'CASH_ADVANCE_TRX_bin'], axis=1)
```


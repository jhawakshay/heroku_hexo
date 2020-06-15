---
title: k-means Clustering of Credit Card Customers
date: 2019-04-29 21:58:47
tags: [k-means, Clustering, Credit Card Marketing, Customer Behaviour]
---

## How can can we leverage Clustering to develop strategies for our Credit Card Customers?

Credit Card is a big business for any Bank. Issuing a Credit card not only a Brand awareness strategy but a revenue generating exercise too. Credit card gives users an extra income to meet their needs which they can repay by next month to the bank without paying an interest.

The tricky part is not everyone thinks like that! Not everyone has the ability to pay his/her dues. Not everyone thinks from a Budget perspective.Not everyone clear dues on time.

All these n number of combinations is where Banks can make money by getting the interest or what we say as a revolving customer. A revolving customer is one who just pays the minimum amount as his/her due so that he is not considered to be missed his Credit Card Payment. So, it is a big opportunity for a Bank to make money but it has to work closely with its Risk as well as Strategy team which finally works with the Marketing team.

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
CREDIT_LIMIT	          1	        0.0

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
Since, there are also outliers in our dataset, we will be creating bins and then use them to do k-means clustering

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

Dropping some of the columns to start with k-means clustering

```python
data_model  = data_n.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'Balance_decile',
       'freq_purchase_decile', 'CC_utilisation', 'TENURE', 'PURCHASES_TRX_bin', 'CASH_ADVANCE_TRX_bin','CC_util_decile'], axis=1)
```

Now, Standardizing the variable for k-means

```python
stand_         = StandardScaler()
data_model_std = stand_.fit_transform(data_model)

random.seed(234)
n_clusters=20
sse=[]
for i in range(1,n_clusters+1):
    kmean= KMeans(i)
    kmean.fit(data_model_std)
    sse.append([i, kmean.inertia_]) 
```

Plotting the ELBOW CURVE to select the number of clusters for our analysis

```python
plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1])
plt.title("Elbow Curve")
```
![Elbow Curve](/images/clustering/elbow.png)

As a general rule, the optimum number of clusters are the ones where the Sum of Squared Distance between Centroid (SSE) is flat and
there is not much change in it

```python
random.seed(234)
kmean= KMeans(8)
kmean.fit(data_model_std)
```

```python
y_kmeans = kmean.predict(data_model_std)
```

```python
data_model['Cluster']       = y_kmeans
data_model_std              = pd.DataFrame(data_model_std)
data_model_std['Cluster']   = y_kmeans
```

Now, Visualizing the clusters to understand if it is possible to combine some clusters and profile them later on

```python
for c in data_model:
    g   = sns.FacetGrid(data_model, col='Cluster')
    g.map(plt.hist, c, color = "red")
```
![](/images/clustering/Facet_grid_1_1.PNG)
![](/images/clustering/Facet_grid_1_2.PNG)
![Seaborn FacetGrid for all the clusters](/images/clustering/Facet_grid_1_3.PNG)


#### Seeing the clusters we try to regroup them on the basis of their variable distributions
##### Cluster 1 and 3 could be combined
##### Cluster 0 and 2 could be combined
##### Cluster 4 and 5 could be combined

```python
data_model["Cluster"].replace({3: 1, 2: 0, 5:4}, inplace=True)
data_model_std["Cluster"].replace({3: 1, 2: 0, 5:4}, inplace=True)
clusters_   = data_model["Cluster"]
```
### Visualizing the clusters in a 2-D axis
For this to happen we would do Principal Component Analysis (PCA) to get two principal Components which will be our X & Y axis

```python
random.seed(32)
pca = PCA()
pca.fit(data_model_std)
```

Selecting the Principal components is based on 80% cumulative variance been explained. As we see, 2 Principal Components are able to 
explain this much of variance

```python
fig = plt.figure(figsize =(12,6))
plt.plot(range(0,12),pca.explained_variance_ratio_.cumsum(), marker ='o', linestyle = "--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
```
![PCA Variance explanation](/images/clustering/img9.png)

### Plotting the Clusters in 2-D

```python
pca = PCA(n_components = 3)
pca.fit(data_model_std)

scores = pca.transform(data_model_std)

x,y = scores[:,0] , scores[:,1]
df_data = pd.DataFrame({'x': x, 'y':y, 'clusters':clusters_})
grouping_ = df_data.groupby('clusters')
```

```python
fig, ax = plt.subplots(figsize=(20, 13))

names = {0: 'Cluster 1', 
         1: 'Cluster 2', 
         4: 'Cluster 3',
         6: 'Cluster 4',
         7: 'Cluster 5'}

for name, grp in grouping_:
    ax.plot(grp.x, grp.y, marker='o', label = names[name], linestyle='')
    ax.set_aspect('auto')

ax.legend()
plt.show()
```

![Initial Clusters](/images/clustering/Initial_clusters.png)
![Final Clusters](/images/clustering/Cluster_3.png)

#### Creating a 3-D Visualization using Plotly

```python
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
```

```python
x,y,z = scores[:,0] , scores[:,1], scores[:,2]

df_data = pd.DataFrame({'x': x, 'y':y, 'z':z, 'clusters':clusters_})
```

```python
# Visualize cluster shapes in 3d.

cluster1=df_data.loc[df_data['clusters'] == 0]
cluster2=df_data.loc[df_data['clusters'] == 1]
cluster3=df_data.loc[df_data['clusters'] == 4]
cluster4=df_data.loc[df_data['clusters'] == 6]
cluster5=df_data.loc[df_data['clusters'] == 7]


scatter1 = dict(
    mode = "markers",
    name = "Cluster 1",
    type = "scatter3d",    
    x = cluster1.to_numpy()[:,0], y = cluster1.to_numpy()[:,1], z = cluster1.to_numpy()[:,2],
    marker = dict( size=2, color='green')
)
scatter2 = dict(
    mode = "markers",
    name = "Cluster 2",
    type = "scatter3d",    
    x = cluster2.to_numpy()[:,0], y = cluster2.to_numpy()[:,1], z = cluster2.to_numpy()[:,2],
    marker = dict( size=2, color='blue')
)
scatter3 = dict(
    mode = "markers",
    name = "Cluster 3",
    type = "scatter3d",    
    x = cluster3.to_numpy()[:,0], y = cluster3.to_numpy()[:,1], z = cluster3.to_numpy()[:,2],
    marker = dict( size=2, color='red')
)

scatter4 = dict(
    mode = "markers",
    name = "Cluster 4",
    type = "scatter3d",    
    x = cluster4.to_numpy()[:,0], y = cluster4.to_numpy()[:,1], z = cluster4.to_numpy()[:,2],
    marker = dict( size=2, color='orange')
)

scatter5 = dict(
    mode = "markers",
    name = "Cluster 5",
    type = "scatter3d",    
    x = cluster5.to_numpy()[:,0], y = cluster5.to_numpy()[:,1], z = cluster5.to_numpy()[:,2],
    marker = dict( size=2, color='yellow')
)


################## Clusters  ##############

cluster1 = dict(
    alphahull = 5,
    name = "Cluster 1",
    opacity = .1,
    type = "mesh3d",    
    x = cluster1.to_numpy()[:,0], y = cluster1.to_numpy()[:,1], z = cluster1.to_numpy()[:,2],
    color='green', showscale = True
)
cluster2 = dict(
    alphahull = 5,
    name = "Cluster 2",
    opacity = .1,
    type = "mesh3d",    
    x = cluster2.to_numpy()[:,0], y = cluster2.to_numpy()[:,1], z = cluster2.to_numpy()[:,2],
    color='blue', showscale = True
)
cluster3 = dict(
    alphahull = 5,
    name = "Cluster 3",
    opacity = .1,
    type = "mesh3d",    
    x = cluster3.to_numpy()[:,0], y = cluster3.to_numpy()[:,1], z = cluster3.to_numpy()[:,2],
    color='red', showscale = True
)

cluster4 = dict(
    alphahull = 5,
    name = "Cluster 4",
    opacity = .1,
    type = "mesh3d",    
    x = cluster4.to_numpy()[:,0], y = cluster4.to_numpy()[:,1], z = cluster4.to_numpy()[:,2],
    color='orange', showscale = True
)

cluster5 = dict(
    alphahull = 5,
    name = "Cluster 5",
    opacity = .1,
    type = "mesh3d",    
    x = cluster5.to_numpy()[:,0], y = cluster5.to_numpy()[:,1], z = cluster5.to_numpy()[:,2],
    color='yellow', showscale = True
)

layout = dict(
    title = '3D visulization of Clusters',
    scene = dict(
        xaxis = dict( zeroline=True ),
        yaxis = dict( zeroline=True ),
        zaxis = dict( zeroline=True ),
    )
)
fig = dict( data=[scatter1, scatter2, scatter3, scatter4, scatter5, cluster1, cluster2, cluster3, cluster4, cluster5], layout=layout )
# Use py.iplot() for IPython notebook
plotly.offline.iplot(fig, filename='mesh3d_sample')

```
![3-D Visualization with Plotly](/images/clustering/newplot.png)
<img src="/images/clustering/ezgif.com-video-to-gif.gif" style="width:400;height:400;">

With this image you won't be able to interact with the plot but in my github code (Link given at the end) you will be able to easily interact with the 3-D visualization. Plotly is so powerful!


Now comes the most important part of the project, Consumption. No matter how great analysis is been done but if one can't consume the 
output, there is no point wasting time. We will be profiling our customers/clusters so that marketing and risk team can easily use 
them to generate more business and mitigate risk.

```python
for c in data_model:
    g   = sns.FacetGrid(data_model, col='Cluster')
    g.map(plt.hist, c, color = "red")
```
![](/images/clustering/facet_grid_2_1.PNG)
![](/images/clustering/facet_grid_2_2.PNG)
![](/images/clustering/facet_grid_2_3.PNG)
![](/images/clustering/facet_grid_2_4.PNG)
![](/images/clustering/facet_grid_2_5.PNG)
![](/images/clustering/facet_grid_2_6.PNG)
![](/images/clustering/facet_grid_2_7.PNG)
![](/images/clustering/facet_grid_2_8.PNG)
![](/images/clustering/facet_grid_2_9.PNG)
![](/images/clustering/facet_grid_2_10.PNG)
![Profiling of the final clusters](/images/clustering/facet_grid_2_11.PNG)

As we can see from the above image, following is the profile of the 5 Clusters:

#### Cluster 0: Who do not purchase but have good credit limit. Also miss payments 
#### Cluster 1: Who have a good balance, make average purchases and do make payments
#### Cluster 4: Who buy frequntly and have a high credit limit
#### Cluster 6: Who buy very small, keeps low balance but frequently pay dues
#### Cluster 7: Who buy in installments only

Risk team will be interested in doing more analysis on the customers which are in Cluster0.
Marketing team will be interested in cross-selling offers from Merchants to Cluster4 and Cluster7
Bank won't be able to do much business with Cluster6 and will just keep a watch that they pay their dues

With such a small data, we are able to certainly add value to our analysis. Clustering is such an iterative but interesting exercise.
Once, you find these clusters and Marketing team works to implement, you can see results in 2-3 months
[Github code](https://github.com/jhawakshay/solving_fintech_with_ML)

Please write to me at akshayjhawar.nitj@gmail if you have more views

Until then, keep looking which cluster you belong to.

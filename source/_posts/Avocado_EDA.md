---
title: Avocado Prices Data Visualization
date: 2020-01-20 21:58:47
tags: [EDA, Charts, Matplotlib, Data Visualization]
---

## Data Visulization on Avocado Data

I was more keen of analyzing data on Avocados and find some interesting trends. So, below is what I am worked on the EDA for Avocado data.

### Importing Python Libraries

```python
#Importing datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import calendar
import plotly.express as px
import seaborn as sns
import warnings
from plotnine import *
from plotnine.data import *
import scipy as sp
from scipy.interpolate import interp1d
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
```
### Reading Avocado Data

```python
raw_data = pd.read_csv('E:/avocado-prices/avocado.csv')
```
Here is the link to download data from Kaggle [Avocado](https://www.kaggle.com/neuromusic/avocado-prices) 

### Exploring the Data

```python
raw_data.describe()
```

### Change format of Date Columns and Create columns for Year and Month

```python
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data         = raw_data.sort_values('Date')
raw_data['Year']         = pd.DatetimeIndex(raw_data['Date']).year
raw_data['Month']        = pd.DatetimeIndex(raw_data['Date']).month
raw_data['Month_Name']   = raw_data['Month'].apply(lambda x: calendar.month_abbr[x])
```

### Histograms for Average Price

```python
fig = plt.figure(figsize = (9,6))
plt.hist(raw_data['AveragePrice'], bins = 30, alpha = 0.4, color='green', density=True)
plt.title('Histogram plot of Average Prices')
plt.grid(b=None)
```

![Distribution of Average Price of Avocado](/images/Avocado/hist1.png)

```python
fig, ax =plt.subplots(1,2,figsize=(13,6))
ax0, ax1 = ax.flatten()
x = raw_data[raw_data['type'] =='organic']
y = raw_data[raw_data['type'] !='organic'] 

ax0.hist(x['AveragePrice'], bins = 30, alpha = 0.9, color='blue', density=True)
ax1.hist(y['AveragePrice'], bins = 30, alpha = 0.9, color='orange', density=True)

ax0.set_title("Organic")
ax1.set_title("Conventional")

ax0.set_ylabel('Average Price')
```
![Distribution of Average Price of diff Avocado](/images/Avocado/hist2.png)
We see that Conventional Avocados are a little right skewed and have peak of around 1.65

```python
sns.violinplot(y='AveragePrice',x='type',data=raw_data)
```
![Box Plot of diff Avocado on Average Price](/images/Avocado/hist3.png)

### Price trend across year for different types of Avocados

```python
ig, ax =plt.subplots(1,2,figsize=(20,6))
ax0, ax1 = ax.flatten()
## Organic
X = raw_data[raw_data['type'] =='organic']
groupBy1_price = X.groupby('Date',as_index=False).mean()
XX = groupBy1_price['Date']
YY = groupBy1_price['AveragePrice']

ax0.plot(XX,YY, color = "blue")
ax0.set_title('Organic')
ax0.set_ylabel('Average Price')
ax0.set_xlabel('Date')

## Conventional
Y = raw_data[raw_data['type'] !='organic'] 
groupBy2_price = Y.groupby('Date',as_index=False).mean()
XX = groupBy2_price['Date']
YY = groupBy2_price['AveragePrice']

ax1.plot(XX,YY, color = 'orange')
ax1.set_title('Conventional')
ax1.set_ylabel('Average Price')
ax1.set_xlabel('Date')
```

![Box Plot of diff Avocado on Average Price](/images/Avocado/hist4.png)
We see some interesting trends here:
1. Average Prices go down in mid of 2017 from its peak
1. Organic Avocados have followed a real sinusoidal graph with bigger peaks


### Plotting volume with Average Prices
```python
fig, ax =plt.subplots(2,1,figsize=(40,40))
ax0, ax1 = ax.flatten()

ax2 = ax0.twinx()  # set up the 2nd axis
ax3 = ax1.twinx()



groupBy1_Vol = X.groupby('Date',as_index=False).sum()
XX = groupBy1_Vol['Date']
YY = groupBy1_Vol['Total Volume']
YYY = groupBy1_price['AveragePrice']


ax0.bar(XX, YY,width=4, color='blue', alpha =0.4)
ax0.ticklabel_format(style='plain',axis='y')
ax0.set_title('Volume and Price Trend of Organic Avocados', fontsize = 40)
ax0.set_xlabel('Date',fontsize = 40)
ax0.set_ylabel('Volume',fontsize = 40)
ax0.tick_params(axis='both', which='major', labelsize=30)
ax2.plot(XX,YYY, color = 'red', linewidth=3)
ax2.set_ylabel('Average Price',fontsize = 40)
ax2.tick_params(labelsize =30)

groupBy2_Vol = Y.groupby('Date',as_index=False).sum()
XX = groupBy2_Vol['Date']
YY = groupBy2_Vol['Total Volume']
YYY = groupBy2_price['AveragePrice']

ax1.bar(XX, YY,width=8, color='orange', alpha = 0.3)
ax1.ticklabel_format(style='plain',axis='y')
ax1.set_title('Volume and Price Trend of Conventional Avocados', fontsize=40)
ax1.set_xlabel('Date',fontsize = 40)
ax1.set_ylabel('Volume',fontsize = 40)
ax1.tick_params(axis='both', which='major', labelsize=30)
ax3.plot(XX,YYY, color = 'red', linewidth=3)
ax3.set_ylabel('Average Price',fontsize = 40)
ax3.tick_params(labelsize =30)
```
![Plotting Volume vs Price for different Avocados](/images/Avocado/hist5.png)

If we closely look at the above chart:
1. We see a clear relationship of Demand and Supply. As the Volumne increase, Prices go down
2. 2017 end, the Prices drop as the volume increases

```python
fig, ax =plt.subplots(1,2,figsize=(17,4))
sns.violinplot(y='AveragePrice',x='Year',data=raw_data, ax=ax[0])
sns.violinplot(y='AveragePrice',x='Month_Name',data=raw_data, ax=ax[1])
fig.show()
```
![Box Plots by Year and Month](/images/Avocado/hist6.png)

### Distribution of Average Price by Year

```python
fig, ax =plt.subplots(2,2,figsize=(13,6))
ax0, ax1, ax2, ax3 = ax.flatten()

x1 = raw_data[raw_data['Year'] ==2015]
x2 = raw_data[raw_data['Year'] ==2016]
x3 = raw_data[raw_data['Year'] ==2017]
x4 = raw_data[raw_data['Year'] ==2018]

ax0.hist(x1['AveragePrice'], bins = 30, alpha = 0.9, color='blue', density=True)
ax1.hist(x2['AveragePrice'], bins = 30, alpha = 0.9, color='orange', density=True)
ax2.hist(x3['AveragePrice'], bins = 30, alpha = 0.9, color='Green', density=True)
ax3.hist(x4['AveragePrice'], bins = 30, alpha = 0.9, color='red', density=True)

ax0.set_title("Average Price Distribution for Year 2015", fontsize=10)
ax1.set_title("Average Price Distribution for Year 2016", fontsize=10)
ax2.set_title("Average Price Distribution for Year 2017", fontsize=10)
ax3.set_title("Average Price Distribution for Year 2018", fontsize=10)

ax0.set_ylabel("Average Price")
ax1.set_ylabel("Average Price")
ax2.set_ylabel("Average Price")
ax3.set_ylabel("Average Price")

```
![Box Plots by Year and Month](/images/Avocado/hist7.png)

We see in the above chart:
1. Average Prices are right skewed in the year of 2015 and 2016
2. Price peak is more for 2015 and 2018 compared to 2016 and 2017


### Analyzing prices by months and type of Avocados

```python
fig, ax =plt.subplots(1,4,figsize=(20,6))
ax0, ax1, ax2, ax3 = ax.flatten()
Year  = [2015,2016,2017,2018]

## Organic

for i in range(0,4):
    X = raw_data[(raw_data['type'] =='organic') & (raw_data['Year'] ==Year[i])]
    groupBy1_price = X.groupby('Month_Name',as_index=False).mean()
    groupBy1_price = groupBy1_price.sort_values(by=['Month'])
    XX = groupBy1_price['Month_Name']
    YY = groupBy1_price['AveragePrice']
    
    ax[i].plot(XX,YY, color = "blue", marker='o', linestyle='--')
    ax[i].set_title("Organic Avocados in {}".format(Year[i]))
    ax[i].set_ylabel('Average Price')
    ax[i].set_xlabel('Date')
    


fig, axx =plt.subplots(1,4,figsize=(20,6))
axxo, axx1, axx2, axx3 = axx.flatten()
Year  = [2015,2016,2017,2018]

## Conventional

for i in range(0,4):
    X = raw_data[(raw_data['type'] =='conventional') & (raw_data['Year'] ==Year[i])]
    groupBy1_price = X.groupby('Month_Name',as_index=False).mean()
    groupBy1_price = groupBy1_price.sort_values(by=['Month'])
    XX = groupBy1_price['Month_Name']
    YY = groupBy1_price['AveragePrice']
    
    axx[i].plot(XX,YY, color = "orange", marker='o', linestyle='--')
    axx[i].set_title("Conventional Avocados in {}".format(Year[i]))
    axx[i].set_ylabel('Average Price')
    axx[i].set_xlabel('Date')
```
![Price trend by Year and type of Avocados](/images/Avocado/hist8.png)
![Price trend by Year and type of Avocados](/images/Avocado/trend.png)
### Check Price Volatility for each year

```python
fig, ax =plt.subplots(1,3,figsize=(18,6))
ax0, ax1, ax2 = ax.flatten()
Year  = [2015,2016,2017]

## Organic

for i in range(0,3):
    X = raw_data[(raw_data['type'] =='organic') & (raw_data['Year'] ==Year[i])]
    X = X[['Month_Name', 'Month','AveragePrice']]
    groupBy1_price = X.groupby(['Month_Name', 'Month'],as_index=False)[['AveragePrice']].agg(np.std, ddof=0)
    groupBy1_price = groupBy1_price.sort_values(by=['Month'])
    XX = groupBy1_price['Month_Name']
    YY = groupBy1_price['AveragePrice']
    
    ax[i].plot(YY,XX, color = "blue", marker='o', linestyle = 'None', markersize=12)
    ax[i].set_title("Organic Avocados in {}".format(Year[i]))
    ax[i].set_ylabel('Month')
    ax[i].set_xlabel('Standard Deviation')
    ax[i].yaxis.grid()


fig, axx =plt.subplots(1,3,figsize=(20,6))
axxo, axx1, axx2 = axx.flatten()
Year  = [2015,2016,2017]

## Conventional

for i in range(0,3):
    X = raw_data[(raw_data['type'] =='conventional') & (raw_data['Year'] ==Year[i])]
    X = X[['Month_Name', 'Month','AveragePrice']]
    groupBy1_price = X.groupby(['Month_Name', 'Month'],as_index=False)[['AveragePrice']].agg(np.std, ddof=0)
    groupBy1_price = groupBy1_price.sort_values(by=['Month'])
    XX = groupBy1_price['Month_Name']
    YY = groupBy1_price['AveragePrice']
    
    axx[i].plot(YY, XX, color = "orange", marker='o', linestyle='None',markersize=12)
    axx[i].set_title("Conventional Avocados in {}".format(Year[i]))
    axx[i].set_ylabel('Month')
    axx[i].set_xlabel('Standard Deviation')
    axx[i].yaxis.grid()
```
![Price Volatility](/images/Avocado/hist9.png)
![Price Volatility](/images/Avocado/trend1.png)

We can dig more deeper to find interesting trends for Seasonal changes, regions selling high volume of Avocados orsome more volume
trends.

[Github Code link](https://github.com/jhawakshay/Data_Visualization/tree/master/Python)
Please write to me at akshayjhawar.nitj@gmail.com if you have more ideas

Until then keep exploring Data!




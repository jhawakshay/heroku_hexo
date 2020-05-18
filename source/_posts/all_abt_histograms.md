---
title: Plotting histograms with matplotlib,seaborn and plotly
date: 2019-10-02 21:58:47
tags: [Data Visualization, Python, Histograms, plotly]
---

## Plotting histograms with various packages

A histogram is an easy yet important data chart which every Data Scientist or an Analyst comes across. It is normally used in the Exploratory 
Data Analysis (EDA) we doe before jumping onto the step of model development. Histograms are used for:

1. Checking the distribution of a variable which helps in checking skewness, kurtosis & outliers 
1. Univariate Analysis for each variable
1. Comparing the distributions after changes to the varible by comparing the histograms on same scale

In the below article, I am trying to show all the different ways to plot histograms:

### Importing libraries and creating the data

```python
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import warnings 
warnings.filterwarnings('ignore')

numValues = 10000
maxValue = 1
skewness = 2   #Negative values are left skewed, positive values are right skewed.

##Initilize variable for skewness
rand = skewnorm.rvs(a = skewness,loc=maxValue, size=numValues)

rand = rand - min(rand)      #Shift the set so the minimum value is equal to zero.
rand = rand / max(rand)      #Standadize all the vlues between 0 and 1. 
rand = rand * maxValue         #Multiply the standardized values by the maximum value.

#Creating a DataFrame
prob_ = pd.DataFrame(rand, columns = ['Prob_'])

## Converting the zero probabilities to nearly zero
prob_['Prob_'] = np.where(prob_['Prob_']==0,0.01,prob_['Prob_'])
```

### Generating histograms using normal function

```python
prob_.hist(column = 'Prob_')
```
![Histogram using .hist()](/images/histograms/hist1.png)

### Putting some arguments into the previou plot

 ```python
 prob_.hist(column = 'Prob_', bins =100, grid = False, figsize = (5,5), color = 'green')
 ```
 ![Histogram using .hist() and arguments](/images/histograms/hist2.png)
 
 ### Generating histograms using matplotlib
 
 ```python
fig = plt.figure(figsize = (5,5))
plt.hist(prob_['Prob_'],bins = 30,density=False, color = 'blue', alpha=0.9)
plt.show()
 ```
![Histogram using matplotlib](/images/histograms/hist3.png)

### Adding other arguments like titles, density

```python
#Plot histogram to check skewness
fig = plt.figure(figsize = (5,5))
plt.hist(prob_['Prob_'],bins = 30,density=True, color = 'green', alpha=0.9)
plt.xlabel("Probability", fontsize = 10)
plt.ylabel("Density")
plt.xticks(fontsize =10, rotation=45)
plt.title("Density Plot of Probability", fontsize = 20)
plt.show()
```
![Histogram using matplotlib and other options](/images/histograms/hist4.png)

### Generating histograms using seaborn

```python
fig = plt.figure(figsize = (5,5))
ax = sns.distplot(prob_['Prob_'], hist=True, kde=False, 
             bins=100, color = 'blue')

ax.set_xlabel("Probability")
ax.set_ylabel("Count")
ax.set_title("Histogram of Probability")
```

![Histogram using seaborn](/images/histograms/hist5.png)

### Adding a density plot using Gaussian curve

```python
fig = plt.figure(figsize = (5,5))
ax = sns.distplot(prob_['Prob_'], hist=True, kde=True,
                        bins=100, color = 'blue',hist_kws={'edgecolor':'black'},
                         kde_kws={'linewidth': 4})

ax.set_xlabel("Probability")
ax.set_ylabel("Count")
ax.set_title("Histogram of Probability")
```
![Histogram using seaborn and adding density plot](/images/histograms/hist6.png)

## Comparing two distributions with two or more histograms in one plot

This is something I have pre-dominantly used in my post model analysis when I do monitoring of models and see how model is behaving 
after some months of deployment

```python
## This code is just to calcualte Scores and see distribution of these Scores
Base_Score = 700
pdo        = 70
Good_Bads  = 50

## Creating a function to calculate a Score
def score_(x):
    score_ = Offset - Factor * np.log(x)
    return score_

## Score 1
Factor          = pdo/np.log(2)
Offset          = Base_Score - Factor * np.log(Good_Bads)
prob_['Score']  = score_(prob_['Prob_'])

## Score 2
pdo        = 120
Factor     = pdo/np.log(2)
Offset     = Base_Score - Factor * np.log(Good_Bads)
prob_['Score_2']  = score_(prob_['Prob_'])
```

### Using matplotlib
```python
fig = plt.figure(figsize=[10,8])
x = prob_['Score']
y = prob_['Score_2']
n, bins, patches = plt.hist([x, y], bins=100, density=True)
plt.legend({'Score':x, 'Score 2':y})
plt.xlabel("Score")
plt.ylabel("Density")


```

![Comparing Histograms using matplotlib](/images/histograms/hist7.png)

### Using seaborn

We need to append all the various Score variables into one and then create a flag for these. You can add as many as histograms by 
running a loop then. We can also add mean, median or mode to the histogram and compare more

```python
## Creating two dataframe with an indicator showing different scores
#prob_           = prob_.drop(['Score_2'], axis = 1)
prob_['Type']   = 'Score1'
prob__          = prob_.copy()                     ## Creating a copy of the dataframe
prob__          = prob_.drop(['Score', 'Type'], axis = 1)  ## Dropping the other columns

## Calculating the score
pdo        =  50
Factor     = pdo/np.log(2)
Offset     = Base_Score - Factor * np.log(Good_Bads)
prob__['Score']  = score_(prob__['Prob_'])
prob__['Type']   = 'Score2'

## Appending both the table
full_            = prob_.append(prob__)
```

```python
type = ['Score1', 'Score2']

fig = plt.figure(figsize = (15,8))
for scenario in type:
    subset = full_[full_['Type']==scenario]
    
    mean   = np.mean(subset['Score'])
    median   = np.median(subset['Score'])
    
    ax = sns.distplot(subset['Score'], hist=True, kde=True, 
             bins=100, 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2}, label= scenario)
   
    ax.axvline(mean, color='red', linestyle='--', linewidth=2)
    ax.axvline(median, color='green', linestyle='--', linewidth=2)
    ax.set_xlabel("Score Range")
    ax.set_title("Density Plot of Behaviour Scores", fontsize=15)
    plt.legend()
```
![Comparing Histograms using seaborn](/images/histograms/hist8.png)

### Plotting just the Gaussian curve with the mean & median

```python
type = ['Score1', 'Score2']

fig = plt.figure(figsize = (15,8))
for scenario in type:
    subset = full_[full_['Type']==scenario]
    
    mean   = np.mean(subset['Score'])
    median   = np.median(subset['Score'])
    
    ax = sns.distplot(subset['Score'], hist=False, kde=True, 
             bins=100, 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1}, label= scenario)
   
    ax.axvline(mean, color='red', linestyle='--', linewidth=2)
    ax.axvline(median, color='green', linestyle='--', linewidth=2)
    ax.set_xlabel("Score Range")
    ax.set_title("Density Plot of Behaviour Scores", fontsize=15)
    plt.legend()
```
![Comparing Histograms using seaborn and adding mean & median vertical lines](/images/histograms/hist9.png)
![Comparing Histograms with only Density curves](/images/histograms/hist10.png)
### Drawing pair plot using seaborn

This is helpful when we want to see distributions of all the variables at one go and how do they compare with each other. A powerful
line of code which gives immense value addition

```python
ax = sns.pairplot(full_, hue="Type")
ax.fig.set_size_inches(12,8)
```
![Pair Plots using seaborn](/images/histograms/hist10.png)

### Generating charts with plotly

Plotly is a more advanced and highly interactive open source plotting library which can be used in your daily projects. It is like a
dynamic Dashboard but still just a chart.

```python
fig = px.histogram(prob_, x="Score", nbins=200)
fig.update_layout(
    title="Histogram of Score",
    xaxis_title="Score",
    yaxis_title="Count")
fig.show()
```
![Interactive Histograms using plotly](/images/histograms/newplot.png)

### Generating two plots

```python
fig = px.histogram(full_, x="Score", color="Type")
fig.update_layout(
    title="Histogram of Score",
    xaxis_title="Score",
    yaxis_title="Count")
fig.show()
```
![ComparingHistograms using plotly](/images/histograms/newplot1.png)

And there is much more you can do.

I hope you liked the basics and interesting things you can do with histograms.

Please free feel to write to me on akshayjhawar.nitj@gmail.com if you have any suggestions








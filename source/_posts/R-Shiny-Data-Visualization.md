---
title: Using smbinning & RShiny to automate WoE & IV
date: 2019-12-23 21:58:47
tags: [R Shiny, Data Visualization, Fintech, Banking]
---


## Developing Scorecards for your portfolio
###### Smbinning Package for Scorecard Analysis

###### 
In this article, I am going to talk about the concept of Weight of Evidence (WoE) & Information Value (IV) which is used in predictive
modelling. Woe & IV is a conceptual exercise which is an important step while reducing the variables before the step of statistical 
modelling. Also, it brings linearity to a non-linear problem.

I have used WoE & IV techniques while developing a lot of predictive scorecards for Banking and Fintech firms. A traditinal scorecard
is a kind of a classifier develop on Logistic regression. In Banking, we separate goods & bads customers.

The article will proceed in the following way:

* WoE & IV formula
* Importance of WoE & IV
* Automation of calculations of WoE & IV on RShiny

**Weight of Evidence (WoE)**
The weight of evidence tells the predictive power of an independent variable in relation to the dependent variable. It was 
involved from credit scoring world, so it a measure to separate between goods and bads.


**Banking & FinTech**
```python
Weight of Evidence  = ln(Distribution of Goods/Distribution of Bads)

Distribution of Goods = %Goods in the total population
Distribution of Bads  = %Bads in the total population
```

**General**
```python

Weight of Evidence  = ln(% of Non-Events/% of Events)

```

**Steps to calculate WoE**
1. For a continous variable, split data into 10 groups (it can be less as per the distribution)
1. Calculate the number of events & non-events as per all the groups
1. Calculate the % of events and % of non-events of each group
1. Calculate the WoE by taking the natural log of division of % of non-events and % events

**An example**
There are 300 customers in a portfolio of Credit Cards where the number of Defaults (Events) are 40 and number of 
Non-Defaults (Non-Events) are 260. A WoE table will be like this:

| Bins | Defaults | Non-Defaults | Default % | Non - Default %| WoE |
|------|----------|--------------|-----------|----------------|-----|
|10-30 |   5      |       50     |    12.5%  |      19.23%    | 0.43|
|31-60 |  15      |       80     |    37.5%  |      30.76%    |-0.19|
|61-90 |  20      |      130     |    50%    |      50%       | 0.0 |
|Total |  40      |      260     |    100%   |      100%      |     |


**Benefits of WoE**
1. WoE helps to treat outliers. When we create bins for a variable, Outliers can be provided a WoE value for that respective
   bin/group.
2. It can also handle missing values in the dataset
3. It helps to find strict linear relationship with the lof odds. 



**Information Value**
It is a calue which tells us the importance of each variable in a predictive model. It is a way to rank variables in a 
model. The formula for IV is:

**Banking & FinTech**
```python
IV = (% of Goods - % Bads) * WoE
```

**General**

```python
IV = (% of Non-Events - % Events) * WoE
```

| Bins | Defaults | Non-Defaults | Default % | Non - Default %| WoE | IV |
|------|----------|--------------|-----------|----------------|-----|----|
|10-30 |   5      |       50     |    12.5%  |      19.23%    | 0.43|0.02|
|31-60 |  15      |       80     |    37.5%  |      30.76%    |-0.19|0.01|
|61-90 |  20      |      130     |    50%    |      50%       | 0.0 |0.0 |
|Total |  40      |      260     |    100%   |      100%      |     |0.03|

**The IV of the variable is 0.03**

**Some Important facts about IV**
1. As there is an increase in the number of bins, there is an increase in the value of IV
2. IV is more specific to Logistic Regression.

To arrive at the optimum binning of one variable and calculate the IV, it is a repetitive task. One needs to first find the 
optimum bins, calculate WoE & then re-group or create more groups, and again calculate WoE.

Now suppose if there are 100 variables and you have to calculate WoE & IV for each of the variable. It requires a lot of time
& effort. There is where I tried developing a **RShiny Dashbaord** using **smbinning package** which automates the whole process 
and gives results instantaneusly.

**Dashboarding on RShiny**

RShiny is a powerful open source dashboarding tool which can be used to automate processes, create infographic dashboards,
drive value analysis and much more. There are important widgets like file input, Slider, Dropwdowns and on top of them all
the packages that exist in R can be used and integrated with RShiny.

![IV Image](/source/images/IV.png)



---
title: Using smbinning & RShiny to automate WoE & IV
date: 2020-04-06 21:58:47
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

```python

Weight of Evidence  = **ln(Distribution of Goods/Distribution of Bads)**

```

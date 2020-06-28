---
title: The mother of all optimization algorithms in Machine Learning
date: 2019-08-1 21:58:47
tags: [Gradient Descent, Optimization, Machine Learning]
---

Everyone knows about Gradient Descent but nobody talks about it so often. Why is that behaviour? Well, running a ML algorithm in R or Python is really easy and by that what I mean is once you have prepared the Data, you just need to get the arguments correct and the data in the correct format. For example, XgBoost accepts data in XgB Matrix or One-hot encoding while Random Forest can accept Data as is. 

What next? You run the algorithm and you get the required results. Forget about Machine Learning Algorithms, even Linear and Logistic regression work on an optimising algorithm. That's why I say Gradient Descent : The mother of all optimization algorithms. Undoubtedly, this algorithm is at the backend of all the modelling techniques.

The earlier you get the basics right, the better you will be able to optimize your hyperparameters and modelling output.

In this article, I am going to talk about three things:
1. Theory of Gradient Descent
1. Practical implementation of Gradient Descent
1. What next if I understand and perfect the above two steps? Am I just giving gyaan or it can really help to be a good Data Scientist

Let's begin

**Gradient Descent**
Before beginning about Gradient Descent I would like to talk about Optimization. Optimization is needed for every real world problem so that if we are finding a solution, the solution not only has to solve the problem but it should be the best one maybe on metrics like Time, Cost, effectiveness. An optimizing solution in logistics could be Finding the best route from Point A to Point B which is what Google Maps does by optimizing time.

In Machine Learning, Optimization is basically working on a solution that works the best on even the unseen data. We do this all over again by optimizing our solution on Training data and then checking on the unseen data (Test data).

Coming back to **Gradient Descent**
Gradient Descent is an algorithm where we try to “walk” in a direction so the function decreases until we no-longer can. This is the most easy explanation in layman terms. Now, let's see it visually. Suppose, there is a stuntman (GD as Gradient Descent) and riding your bicycle in a **U** terrain. He is finishing with his stunt and wants to reach to the safest (risk-free) point in the terrain

There are three images which will help understand what I am talking about. In the first image, the stuntman GD is at the top of the terrain and he starts coming down. While the risk (Steepest) is the most at the top, the good part is that the risk (Steepness) is decreasing. So, he knows he is on the correct path to be risk-free.



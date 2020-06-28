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

**Theory of Gradient Descent**
Before beginning about Gradient Descent I would like to talk about Optimization. Optimization is needed for every real world problem so that if we are finding a solution, the solution not only has to solve the problem but it should be the best one maybe on metrics like Time, Cost, effectiveness. An optimizing solution in logistics could be Finding the best route from Point A to Point B which is what Google Maps does by optimizing time.

In Machine Learning, Optimization is basically working on a solution that works the best on even the unseen data. We do this all over again by optimizing our solution on Training data and then checking on the unseen data (Test data).

Coming back to **Gradient Descent**
Gradient Descent is an algorithm where we try to “walk” in a direction so the function decreases until we no-longer can. This is the most easy explanation in layman terms. Now, let's see it visually. Suppose, there is a stuntman (GD as Gradient Descent) and riding your bicycle in a **U** terrain. He is finishing with his stunt and wants to reach to the safest (risk-free) point in the terrain

There are three images which will help understand what I am talking about. In the first image, the stuntman GD is at the top of the terrain and he starts coming down. While the risk (Steepest) is the most at the top, the good part is that the risk (Steepness) is decreasing. So, he knows he is on the correct path to be risk-free.

![Image1](/images/Gradient Descent/Pic1.png)

Next, he reaches to the middle of the terrain but he still feels he is not totally risk-free. But he has again checked that this the right direction as the steepness has decreased a lot.

![Image2](/images/Gradient Descent/Pic2.png)

Finally, in the third image he reached to the point where he is has no risk (Steepness is zero) and the way he has achieved is through moving in the direction where the next step he takes, the steepness decreases. This is what Gradient Descent algorithms does, you reeach a point which has the lowest cost and there is no further movement in increase of cost and the point is known as the **Global Minimum**.

![Image3](/images/Gradient Descent/Pic3.png)

Reducing risk is just a simple analogy to reduce the cost of that function which is nothing but the derivate of the function with respect to parameters.
![Image4](/images/Gradient Descent/Pic4.png)

There are numerous mathematical notation for defining Global Minima but in simple definition Gradient Descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. These parameters could be Beta estimates and Intercept for a linear model (which is what we will implement) or could be weights of a neural net.

**Practical implementation of Gradient Descent**
We take a classical example of Income vs Age and fit a linear line. The idea is not to just fit a line but fit the best line and the best is achieved through minimizing the cost function using Gradient Descent. The cost function in a Linear regression is Sum of Squared Errors.

Import Python Libraries and define the two variables Income and Age. We will fit a line for the Income which is able to predict the Age of an individual.

```python
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
Income = np.array([1000,2000,4000,3000,5000,6000,8000,9000,10000,12000])
Age = np.array([20,23,24,22,26,33,29,30,32,35])
```

Plotting the Income and Age and visualize it
```python
plt.plot(Income,Age, marker = 'o', linestyle="None", color="blue")
plt.title("Age vs Income")
plt.xlabel("Income")
plt.ylabel("Age")
```
![Age Vs Income](/images/Gradient Descent/agevsincome.png)

Next is we define a function which will predict the Age values given a Slope and Intercept of the model.

```python
# Function to predict Y values with a set of intercept and Slope
def predict_(val, a, b):    
    for i in range(1):
        y_pred  = a + b * val[0]
    return y_pred
```

Now comes the important function of Gradient Descent

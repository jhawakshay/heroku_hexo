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

![Image1](/images/GD/Pic11.png)

Next, he reaches to the middle of the terrain but he still feels he is not totally risk-free. But he has again checked that this the right direction as the steepness has decreased a lot.

![Image2](/images/GD/Pic22.png)

Finally, in the third image he reached to the point where he is has no risk (Steepness is zero) and the way he has achieved is through moving in the direction where the next step he takes, the steepness decreases. This is what Gradient Descent algorithms does, you reeach a point which has the lowest cost and there is no further movement in increase of cost and the point is known as the **Global Minimum**.

![Image3](/images/GD/Pic33.png)

Reducing risk is just a simple analogy to reduce the cost of that function which is nothing but the derivate of the function with respect to parameters.
![Image4](/images/GD/Pic44.png)

There are numerous mathematical notation for defining Global Minima but in simple definition Gradient Descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. These parameters could be Beta estimates and Intercept for a linear model (which is what we will implement) or could be weights of a neural net.

**Practical implementation of Gradient Descent**
We take a classical example of Income vs Age and fit a linear line. The idea is not to just fit a line but fit the best line and the best is achieved through minimizing the cost function using Gradient Descent. The cost function in a Linear regression is Sum of Squared Errors.

Import Python Libraries and define the two variables Income and Age. We will try to find the best line fit between X and Y

```python
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
X = np.array([1,2,4,3,5,6,8,9,11,12])
Y = np.array([2,3,4,2,6,3,9,10,12,15])
```

Plotting the X vs Y
```python
plt.plot(X,Y, marker = 'o', linestyle="None", color="blue")
plt.xlabel("X")
plt.ylabel("Y")
```
![X vs Y](/images/GD/graph.png)

Next is we define a function to predict Y when details of Slope and Intercept are given

```python
# Function to predict Y values with a set of intercept and Slope
def predict_(val, a, b):    
    for i in range(1):
        y_pred  = a + b * val[0]
    return y_pred
```

Now comes the important function of Gradient Descent where we iterate and update the parameters by seeing the Error. Following is the detailed explanation of the below code:

1. We start with initializing Intercept and Slope
1. Also, initialize error_sub and Iter_mat to append values of all the iterations and its error difference
1. Next we loop over the iterations fed into the function. In the loop, we predict y by calling the predict_ function & then calculate the error. In the next lines we calculate the Sum of Square Error (SSE) by and add it to the previous error.
1. We also calculate the Sum Diffence which is by how much the error changed from previous step.
1. Next, we update the parameters of linear regression which is Slope and Intercept by subtracting Error * learning rate from the initialized value
1. This loop then runs over the other iteration using only the updarted parameters and again finding the new parameters as per the error

```python
# Function to check error and reiterate by learning rate

def gradient_descent(data, eta, iters,X,y):
    a = 0  # Initialize Intercept
    b = 0  # Initialize Slope
    y_list = [] #Initilize to plot the lines as we increase the iterations
    
    err_sub  = 0.00001  #Initialize a proxy to calcualte Error Difference
    Iter_mat = pd.DataFrame()       #Initialize an array to append all the values for iterations, SSE & Error Difference
    
    for iterss in range(iters):
        sum_error = 0
        
        for val in data:
            ypred = predict_(val, a, b)
            error = ypred - val[-1]
            
            ## Sum of Sqaured Error SSE
            sum_error += error**2
            sum_error  = round(sum_error,4)
            err_chnge  = round((sum_error/err_sub -1),6)*100  # Calculate Error change from last step
            y_list.append(ypred)   
            
            
            ## Here is where we update the parameters of the linear regression; Intercept and Slope
            a          = a - eta * error  
            for k in range(1):
                b      = b - eta * error * val[k]

        plt.plot(X,y, marker = 'o', linestyle="None", color="blue")
        plt.plot(y_list,X, color = "red")
        y_list = []
        combine      = {'Iterations': [iterss],
                             'SSE (Error)' : [sum_error],
                             'Error Change': [err_chnge]}
        combine      = pd.DataFrame(combine, index=None)
        Iter_mat     = Iter_mat.append(combine)

        err_sub    = sum_error
        #print("Iterations :{}  Learning Rate : {}  Error: {} Error_Change:{} ".format(iterss, eta, sum_error,err_chnge))
    plt.show()
    return a, b, iters,sum_error,err_chnge,Iter_mat
```

Now we call both the functions and see how we arrive at the best fit for our problem.

```python
# Calculate coefficients
data = [[1, 2], [2, 3], [4, 4], [3,2], [5, 6],[6, 3],[8, 9], [9, 10], [11, 12],[12, 15]]
eta = 0.0001
iters = 20  #I started running with 3,10,20
coef = gradient_descent(data, eta, iters,X,Y)
print("Intercept is :{} Slope is : {}".format(coef[0], coef[1]))
```

![Lines as per the 3 iterations run](/images/GD/Image1.png)
![Lines as per the 10 iterations run](/images/GD/Image2.png)
![Lines as per the 20 iterations run](/images/GD/Image3.png)

As you see, with the increase in the number of iterations we move closer to a better fit of the points X & Y. Also, the below graph shows the how SSE is deacresing as we are increasing the number of iterations. To be noted, I have also plotted Error change with each iterations and you see after a point Error change is almost flat.

```python
Iter_mat   = coef[5]
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot('Iterations', 'SSE (Error)', data= Iter_mat, color="blue")
ax2.plot('Iterations', 'Error Change', data= Iter_mat, color="green")

plt.title("SSE vs Iterations")
ax1.set_xlabel('Iterations')
ax1.set_ylabel('SSE')
ax2.set_ylabel('Error Change')
```

![SSE vs Iterations](/images/GD/iters1.png)

I had only run 20 iterations to showcase what is actually happening. I have run 50,000 iterations to get to the best fit
![Best fit](/images/GD/Imagef.png)
![SSE vs Iterations](/images/GD/iters50000.png)

Below is the best fit from runnng Gradient Descent
```python
# Final values after running Gradient Descent
Intercept  = coef[0]
Slope      = coef[1]

y_pred     = Intercept + Slope * X

plt.plot(y_pred,X, linestyle="-", color="red")
plt.plot(X,Y, marker = 'o', linestyle="None", color="blue")
plt.title("Best Fit from the algorithm")
```
![Best fit](/images/GD/Finalfit.PNG)

As a check, we will also verfiy our results by running linear regression on our data
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = np.array([1,2,4,3,5,6,8,9,11,12]).reshape(-1,1)
Y = np.array([2,3,4,2,6,3,9,10,12,15])
model.fit(X,Y)
```
![Verifying results with Linear Regression in Skitlearn](/images/GD/Match.PNG)

---
title: The curious case of Accuracy, Precision, Recall and F1 Score
date: 2019-12-29 21:58:47
tags: [Classification, Statistical Modelling, Accuracy, Confusion matrix]
---


## What is it about Accuracy, Precision, Recall and F1 Score?

We as Data Scientist have come across a lot metrics like Accuracy and Precision while developing a statistical model. It could be either a prediction of sales or a classification algorithm, we usually measure or model on Accuracy. But consider this business problem where we need to detect if a person will get cancer or not. For example, if there are 1000 persons in our dataset and there are 20 patients who have cancer. 

So, the cancer rate is around 2% which is rare-event sceanrio. Other types of rare-events are Spam email detection, Defaulters of a loan or detecting frauds on online transactions.

In our example, if we take 1000 patients as our dataset and create a model which predict if a person will have a cancer or not. After following the mdoelling process, we arrive at a confusion matrix as follow:

![Confusion matrix](/images/confusion_1.PNG)

And below is the confusion matrix for the above example where the cancer detection rate is 2%. Suppose, we created a statistical model to detect if a person has a cancer or not and arrive at the following matrix.

![Confusion matrix for Cancer Detection](/images/confusion_2.PNG)

![Accuracy of a model](/images/accuracy.PNG)
Now, if you look at the above matrix and calculate Accuracy of our model, it is 99% as per the above formula. Realistically, the accuracy is like whooping but if you deep-dive, our model has wrongly predicted 5 patients as Non-Cancer whereas they actually had Cancer. This is really risky as losing even a patient to cancer is worrisome not only for the hospital but also for our model.
99% accuracy doesn't do us anything good.

**Trio-pack**

This is why in the case of rare events data or imbalanced data, we should always focus on the three terms which I will introduce below:

![Precison of a model](/images/Precision.PNG)
![Recall of a model](/images/recall.PNG)
![F1 score of a model](/images/F1.PNG)

Let's calculate these three metrics for our model results:

Precision = 10/(5+10) which is 66.67% </br>
Recall    = 10/(5+10) which is also 66.67% </br>
F1 Score  = 66.67% _(coincidently)_

Do you see a drop from Accuracy to Precision, Recall & F1 Score? That's what our model is worth.

**Precision**</br>
It calculates the percentage of actual positives and total predicted positives from the model. This will give you the true picture of our model performance for the rare events as it doesn't include any element of negatives.







---
title: The curious case of Accuracy, Precision, Recall and F1 Score
date: 2019-12-29 21:58:47
tags: [Classification, Statistical Modelling, Accuracy, Confusion matrix]
---


## What is it about Accuracy, Precision, Recall and F1 Score?

We as Data Scientist have come across a lot metrics like Accuracy and Precision while developing a statistical model. It could be either a prediction of sales or a classification algorithm, we usually measure or model on Accuracy. But consider this business problem where we need to detect if a person will get cancer or not. For example, if there are 1000 persons in our dataset and there are 20 patients who have cancer. 

So, the cancer rate is around 2% which is rare-event sceanrio. Other types of rare-events are Spam email detection, Defaulters of a loan or detecting frauds on online transactions.

In our example, if we take 1000 patients as our dataset and create a model which predict if a person will have a cancer or not. After following the mdoelling process, we arrive at a confusion matrix as follow:

![Confusion matrix](/images/confusion_1.png)

And below is the confusion matrix for the above example where the cancer detection rate is 2%. Suppose, we created a statistical model to detect if a person has a cancer or not and arrive at the following matrix.

![Confusion matrix for Cancer Detection](/images/confusion_2.png)

Finally, if you look at this






---
title: The curious case of Accuracy, Precision, Recall and F1 Score
date: 2019-12-29 21:58:47
tags: [Classification, Statistical Modelling]
---


## What is it about Accuracy, Precision, Recall and F1 Score?

We as Data Scientist have come across a lot metrics like Accuracy and Precision while developing a statistical model. It could be either a prediction of sales or a classification algorithm, we usually measure or model on Accuracy. But consider this business problem where we need to detect if a person will get cancer or not. For example, if there are 1000 persons in our dataset and there are 20 patients who have cancer. 

So, the cancer rate is around 2% which is rare-event sceanrio. Other types of rare-events are Spam email detection, Defaulters of a loan or detecting frauds on online transactions.

In our example, if we take 1000 patients as our dataset and create a model which predict if a person will have a cancer or not. After following the mdoelling process, we arrive at a confusion matrix as follows:

                  |Negative||Positive|
                  |--------||--------|
|Actual| |Negative|| 980   |
|------| |--------|
         |Positive|
         |--------|


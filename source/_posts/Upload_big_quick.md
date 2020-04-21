---
title: Quick fix to upload Big flies in RShiny
date: 2019-10-12 21:58:47
tags: [R Shiny, Data Visualization,Hack, Fix, data.table, R Package]
---

## Upload big and quick

In this post I am going to provide you a quick fix to upload big files _(in Gigabytes)_ within seconds. The pre-requisites for these are very basic; I am assuming one know a little RShiny and have written codes on R.

In RShiny, there is a widget **File Input** which is used to upload files from your computer to your Dashboard so that one can do analysis on the same. The widget is a very easy to use with a click and selecting the file but the tricky part about it is _LIMIT SIZE of FILE_. One cannot upload a file of size more than 5MB.

**YES! THIS MUCH ONLY**

While I was working on one of the Dashboard, I was dealing with Gigabytes of Data and faced the first hurdle. I couldn't step ahead before fixing this and did a lot of research around the same but couldn't find a solution.

As Data Scientists or Data Analysts, one needs to find a solution to a problem. Either by hook or crook! Just Kidding!
I am a big fan of data.table package in R. With a single use of this package, my Data wrangling & Analysis becomes really super faster with just few lines of codes. Without wasting much of your time, here is the fix.





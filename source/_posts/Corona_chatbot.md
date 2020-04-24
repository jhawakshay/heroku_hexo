---
title: Creating Chatbot on Coronavrius
date: 2020-04-03 21:58:47
tags: [ChatBot, CoronaVirus, NLP, Natural Language Processing, AI]
---

## Information gathering on Coronavirus from a CHATBOT using nltk and gensim

Cornavirus has it's spread all over the world affecting more than 200 countries of the world, infecting 2,729,274 people and causing 191,614 deaths [source: JOHN HOPKINS UNIVERSITY, https://coronavirus.jhu.edu/map.html]. 

There is a lot of research and studies going to get a vaccine for trating corona. Now, getting a vaccine is a long task while creating awareness about the virus is the foremost need. There are still a lot of people who are not aware of what is Coronavirus and How it is spread? What are the symptoms of Coronavris and what should we do to prevent it?

![Cornavirus Image(source:Bloomberg)](/images/Corona_virus.jpg)

While I was sitting on my nth day of lockdown, my niece asked me a lot of questions on Coronavirus which I had to google, read and answer it right-away. But this entire process took some time and she was fed-up with my late response. So, there is where I thought let's try to create a Chatbot which spreads information on Coronavirus. So, this post is my trial with creating an easy ChatBot using nltk and gensim models.

I have used Doc2vec model from gensim library to train my model along with all the nltk libraries like tokenization, stop words, lemma

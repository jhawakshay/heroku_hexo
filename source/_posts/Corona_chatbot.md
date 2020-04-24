---
title: Creating Chatbot on Coronavrius
date: 2020-04-03 21:58:47
tags: [ChatBot, CoronaVirus, NLP, Natural Language Processing, AI]
---

## Information gathering on Coronavirus from a CHATBOT using nltk and gensim

Cornavirus has it's spread all over the world affecting more than 200 countries of the world, infecting 2,729,274 people and causing 191,614 deaths [source: JOHN HOPKINS UNIVERSITY, https://coronavirus.jhu.edu/map.html]. 

There is a lot of research and studies going to get a vaccine for trating corona. Now, getting a vaccine is a long task while creating awareness about the virus is the foremost need. There are still a lot of people who are not aware of what is Coronavirus and How it is spread? What are the symptoms of Coronavris and what should we do to prevent it?

![Cornavirus Image (source:Bloomberg)](/images/Corona_virus.jpg)

While I was sitting on my nth day of lockdown, my niece asked me a lot of questions on Coronavirus which I had to google, read and answer it right-away. But this entire process took some time and she was fed-up with my late response. So, there is where I thought let's try to create a Chatbot which spreads information on Coronavirus. So, this post is my trial with creating an easy ChatBot using nltk and gensim models.

I have used Doc2vec model from gensim library to train my model along with all the nltk libraries like tokenization, stop words, removal of special characters. Here, I will show all the steps on how to create a simple CHATBOT.

**Import libraries**

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import random
import pandas as pd
import numpy as np
import warnings
import requests
import re
from string import punctuation
warnings.filterwarnings('ignore')
```

**Import and update gensim**

```python
pip install --upgrade gensim
```
**Import Doc2vec from gensim

```python
import gensim
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import multiprocessing
import os
```

**Import the file**

```python
f = open("E:\Blog\Chatbot\Covid_new_text.txt", 'r', errors = "ignore")
raw = f.read()
raw = raw.lower()
```

**Remove Special Characters from the Document**
```python
review_text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", raw )
review_text = re.sub(r"\'s", " 's ", review_text )
review_text = re.sub(r"\'ve", " 've ", review_text )
review_text = re.sub(r"n\'t", " 't ", review_text )
review_text = re.sub(r"\'re", " 're ", review_text )
review_text = re.sub(r"\'d", " 'd ", review_text )
review_text = re.sub(r"\'ll", " 'll ", review_text )
#review_text = re.sub(r"\(", " ", review_text )
#review_text = re.sub(r"\)", " ", review_text )
review_text = re.sub(r"\?", " ", review_text )
review_text = re.sub(r"\s{2,}", " ", review_text )
```

**Tokenizing the document**
Tokenization is the process of splitting a phrase, sentence into words

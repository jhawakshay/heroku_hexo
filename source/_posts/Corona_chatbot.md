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

```python
sent_tokens = nltk.sent_tokenize(review_text)
```
Tokenization is the process of splitting a phrase or a sentence into words/ smaller chunks at the same time giving some punctuations.
If the sentence is, **'This coronavrius has spread all over the world affecting more than 200 countires'**

After Tokenization, the sentence will be broken into **'This' 'coronavirus' 'has' 'spread' 'all' 'over' 'the' 'world' 'affecting' 'more' 'than' '200' 'countries'**. Tokenization is really helpful for machines to read, understand and make meaning out of it.

In the example, we have broken the document into sentence as tokens and not words. Since, we want a sentence as a reply from our chatbot, we will use sent_tokenize from nltk.

**Remove STOP WORDS**

```python
stop_words = set(stopwords.words('english'))
all_sentence =[]
new_sentence = []

for s in range(0,len(sent_tokens)-1):
    word_tokens = word_tokenize(sent_tokens[s])
    for w in word_tokens:
            if w not in stop_words:
                new_sentence.append(w)

    all_sentence.append(new_sentence)
    new_sentence = []
```

In the above code we are removing STOP WORDS from each of our sentence. A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore. Removing STOP WORDS increase the accuracy of our model.

```python
model_sentence = [[' '.join(i)] for i in all_sentence]
```

**Tagging the document**

```python
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(model_sentence)]
```

In the above code we will be using TaggedDocument function from the Doc2Vec which tags each of the sentence and gives its a serial number. The reason I have taken this step is to help me find the similarity and return the tagged sentence  from the ChatBOT. The user response will be taken by the BOT and will pre-process and then it tries to match the text with all the tagged sentences. Whereever it matches the best, the BOT will return the sentence.

**What is Doc2Vec?**
There are two popular methods from gensim; Word2Vec and Doc2Vec. Word2Vec is a well known concept which is used to generate representation vectors out of words. In Word2Vec , each of the word is represented by a vector as any statistical model processes only numerical numbers. For you to understand Doc2Vec, I will try explaining Word2Vec first.

In general, when you like to build some model using words, simply labeling/one-hot encoding them is a plausible way to go. However, when using such encoding, the words lose their meaning. e.g, if we encode Paris as id_4, France as id_6 and power as id_8, France will have the same relation to power as with Paris. We would prefer a representation in which France and Paris will be closer than France and power. 

![Word2Vec functionalities](/images/Word2Vec.png)

There are two algorithms under Word2Vec; Conitnous Bag of Words (CBOW) & Skip-Gram Model.

**CBOW** creates a sliding window around the current word, to predict it from the context (surrounding words). Each of the word is represented as vectors but the words which represents similar names are closed by. In the above fig, three name will be close by;
Infection-Disease = Winter. So, CBOW takes numerous similar words to predict one word

![Word2vec CBOW model](/images/Word2Vec_Rep1.png)

**Skip-Gram** model is exactly the opposite of **CBOW**. It takes one word to predict numerous words (context). It is considered more accurate than CBOW but could be computaionally slower.

Coming to **Doc2Vec**, it creates numerical representation of all document. But the twist here is; it also adds a Paragraph Id to the vectors. The below example will help you to understand what I am trying to explain.

![Doc2Vec PV-DM model](/images/Doc2Vec_Rep1.png)

So, instead of using just the words, Doc2Vec also uses another vector which is always unique to a document. So, while training the words, the document id (a document vector) is also trained. There are also two types of algorithms in Doc2Vec: 
1. PV-DM _Distribution memmory of Paragraph Version_ which is explained in the figure above
2. PV-DBOW _Words version of Paragraph Vector_ which is nothing but the Skip-Gram in Word2Vec.

We also use Tagged document function from Doc2Vec to break each sentence into a unique document and then representing them with a unique vector (I have used this exactly).

The biggets challenge about Doc2Vec is that your document should be in a good order to get good results else one has to pre-process and then train the model. Let's quickly move to the implement the Doc2Vec model.

**Running the Doc2Vec Model**
```python
max_epochs = 400
vec_size = 100
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)

model.build_vocab(documents)
```
epochs    = It is number of iterations to achieve accuracy
vec_size  = Each of the word is represented with 100 numeric vectors
alpha     = It is the learning rate
min_count = Words below the minimum frequency are dropped and not taken into modelling
dm        = dm=1 is PV-DM model while dm=2 is PV-DBOW

The next line of code develops the vocabulary for the model. After, these steps we need to train the model.

```python
for epoch in range(max_epochs):
    #print('iteration {0}'.format(epoch))
    model.train(documents,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")
```

**Initialising the BOT**

```python
greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup")
greeting_responses = ["hey", "hey hows you?", "hello, how you doing", "hello", "Welcome, I am good and you"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
```
We are creating a function which will give random response if a user types greetings like 'hey or 'good morning' 'good evening'. We first make it lower case and then use random.choice from the random library.

**Function to get the reponse from user input**

```python
def response_(user_response):
    chatbot_return=''
    test_data = word_tokenize(user_response.lower())
    v1 = model.infer_vector(test_data)
    
    ## Find similar sentence to the user response
    similar_doc = model.docvecs.most_similar([v1])
    #print(similar_doc)
    if(similar_doc==0):
        chatbot_return = chatbot_return + "Sorry! I don't understand you"
        return chatbot_return
    else:
        #print(int(similar_doc[0][0]))
        retunr_string = documents[int(similar_doc[0][0])][0]
        #print(retunr_string)
        listToStr = ' '.join([str(elem) for elem in retunr_string])
        chatbot_return = chatbot_return + listToStr
        return chatbot_return
  ```
  
  In the above code we take the user input, find it's vector and find a similar document/paragraph id from the syntax _model.docvecs.most_similar_. Now , if there is some similarity the BOT will find all the Tags from the documents which are similar to the user input. The good part is _model.docvecs.most_similar_ automaticaly sorts the tags as per the highest similar to the lowest similar.

The syntax _documents[int(similar_doc[0][0])][0]_ returns the most similar tag id/Document id and hence we get a reply from the BOT.

```python
flag = True
print("ChatBot: I can share some knowlegde about Coronavirus")
while(flag==True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response=='thanks' or user_response =='thank you'):
            flag = False
            print("Chatbot: You are welcome")
    else:
        if(greeting(user_response)!=None):
                print("ChatBot:" + greeting(user_response))
        else:
                print("Chatbot :", end = "")
                print(response_(user_response))
```

Finally, here we are just writing some if-else conditions to get the require response from the BOT and if there is no appropiate response from the user, the BOT can say 'Sorry! I don't undersand you'. Morevoer, if one person would like to end the discussion I will be typing 'Thanks'


---
title: Text analytics using Text Blob
date: 2020-01-20 21:58:47
tags: [Text Analytics, Sentimental Analysis, Text Blob, NLTK]
---


## Text Analytics using Text Blob and predicting the sentiments
In this article, I am trying to analyze twitter data which is majorly related to US Elections, former US President Obama &
about War & ISIS

### Importing Libraries

```python
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import csv
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
nltk.download('stopwords')
%matplotlib inline
```

### Import sentiment analyzer

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
```

### Reading Data
```python
data_ = pd.read_csv("C:/Users/Hp/Downloads/ce7934ac5eaf11ea/dataset/train_file.csv")
```

### Exploring Data
```python
data_.groupby(['Topic']).mean()
```

```python
data_.Topic.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green","blue"])
```
![Pie chart](/images/NLP/Text_BLOB/Pie_chart.png)

### Creating a variable for sentiment classification as Positive and Negative
```python
data_['Sentiment_Title_new'] = ['Positive' if x >= 0 else 'Negative' for x in data_['SentimentTitle']] 

train_pos = data_[ data_['Sentiment_Title_new'] == 'Positive']
train_pos = train_pos['Title']
train_neg = data_[ data_['Sentiment_Title_new'] == 'Negative']
train_neg = train_neg['Title']
```

### Function to draw Word cloud

```python
def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")

## Call function and draw for Positive sentences
wordcloud_draw(train_pos,'white')
```
![Word Cloud for Positive Sentences](/images/NLP/Text_BLOB/pos.png)

```python
print("Negative words")
## Call function and draw for Negative sentences
wordcloud_draw(train_neg)
```
![Word Cloud for Negative Sentences](/images/NLP/Text_BLOB/neg.png)

### Getting Sentiment Scores
#### Data cleaning using stop words

```python
tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in data_.iterrows():
    words_filtered = [e.lower() for e in row.Title.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and not word.startswith('#')
        and not word.startswith('%')
        and not word.startswith('£')
        and not word.startswith('%')             
        and not word.startswith('\x9d')
        and not word.startswith('(')
        and not word.startswith('[')
        and not word.startswith('ú')
        and not word.startswith('~')
        and word !='\x9d'            
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    words_without_stopwords = " ".join(str(x) for x in words_without_stopwords)
    tweets.append((words_without_stopwords))
```

```python
## Adding this cleaner data to the Train_data
data_['Title_cleaned']  = tweets
```

### Start tokenizationa and create a Dictionary
```python
all_words = data_['Title_cleaned'].str.cat(sep=', ')
Dict = dict.fromkeys(all_words,0)
```

### Predict the Polarity of the sentences
```python
## Now using NTLK to predict the polarity of the Title
sid = SentimentIntensityAnalyzer()
def sentiment_ntlk(text):
    try:
        compound_ = sid.polarity_scores(text)
        return compound_['compound']
    except:
        return None

data_['sentiment_ntlk'] = data_['Title_cleaned'].apply(sentiment_ntlk)
```

### Check Distribution of these Polarity Scores
```python
num_bins = 50
plt.figure(figsize =(10,6))
n, bins, patches = plt.hist(data_.sentiment_ntlk, num_bins, facecolor = 'blue', alpha = 0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Polarity')
plt.show()
```
![Distribution of Polarity](/images/NLP/Text_BLOB/hist.png)


Please share your feedback on akshayjhawar.nitj@gmail.com

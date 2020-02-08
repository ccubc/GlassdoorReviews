#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:04:15 2019

@author: chengchen
"""

import pandas as pd
import nltk
from nltk import FreqDist
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import pickle
import string

#%% load subsample of company reviews (10% randomly selected sample of reviews in year 2016)

file = open('/Users/chengchen/glassdoor/data/processed/review_subset.pkl', 'rb')
df = pickle.load(file)
file.close()
print('number of companies in this dataset:', df['company_name'].nunique())
print('number of comments on pros:', df['review_pros'].count()) 
print('number of comments on cons:', df['review_cons'].count())
# 10972 companies, 194150 comments on pros, 194150 comments on cons
df = df.drop_duplicates(subset =['review_pros', 'review_cons'])
# duplicated comments exist --- might be due to the data scraping process
# 163720 comments on pros and cons respectively

#%% Data Preprocessing
#nltk.download('stopwords') # run this one time
def freq_words(x, terms = 30):
    """draw bar plot of top 30 frequent words"""
    all_words = ' '.join([str(text) for text in x])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count': list(fdist.values())})
    
    # selecting top 30 most frequent words
    d = words_df.nlargest(columns = "count", n = terms)
    plt.figure(figsize = (20,5))
    ax = sns.barplot(data = d, x = "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()

# remove stop words
stop_words = stopwords.words('english')
# add punctuations to stop words
for i in string.punctuation:
    stop_words.append(str(i))
    
def remove_stopwords(rev):
    """ This function does the following transformation on input text:
        (1) lower-casing all the text
        (2) remove english stop words
        (3) remove punctuations
    """
    rev = [r.lower() for r in rev]
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new
# remove short words (length < 3)
df['review_pros'] = df['review_pros'].apply(lambda x: ' '.join([w for w in str(x).split() if len(w)>2]))
review_pros = [remove_stopwords(str(r).split()) for r in df['review_pros']]
# make entire text lowercase
review_pros = [r.lower() for r in review_pros]
freq_words(review_pros)

# remove stop words for review_cons
df['review_cons'] = df['review_cons'].apply(lambda x: ' '.join([w for w in str(x).split() if len(w)>2]))
review_cons = [remove_stopwords(str(r).split()) for r in df['review_cons']]
# make entire text lowercase
review_cons = [r.lower() for r in review_cons]
freq_words(review_cons)

# lemmatization: * To further remove noise from the text, let's use lemmatization from the spaCy library. 
# It reduces any given word to its base form thereby reducing multiple forms of a word to a single word.
# !python -m spacy download en # run this one time
nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output
tokenized_review_pros = pd.Series(review_pros).apply(lambda x: str(x).split())
tokenized_review_cons = pd.Series(review_cons).apply(lambda x: str(x).split())
review_pros2 = lemmatization(tokenized_review_pros)
review_cons2 = lemmatization(tokenized_review_cons)
file = open('/Users/chengchen/glassdoor/data/processed/lemma_reviews.pkl', 'wb')
pickle.dump([review_pros2,review_cons2], file)
file.close()
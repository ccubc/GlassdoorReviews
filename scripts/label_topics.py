#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:56:27 2019

@author: chengchen
"""

import pickle
from gensim import corpora
import gensim
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import copy
import numpy as np
import seaborn as sns
import pandas as pd

#%% load files
file = open('/Users/chengchen/glassdoor/data/processed/lemma_reviews.pkl', 'rb')
[review_pros,review_cons] = pickle.load(file)
file.close()
file = open('/Users/chengchen/glassdoor/data/processed/review_pros_dic_mat.pkl', 'rb')
[dictionary, doc_term_matrix] = pickle.load(file)
file.close()
file = open('/Users/chengchen/glassdoor/model/lda_9_pro.pkl','rb')
lda_model = pickle.load(file)
file.close()
file = open('/Users/chengchen/glassdoor/model/lda_8_cons.pkl','rb')
lda_model_cons = pickle.load(file)
file.close()
file = open('/Users/chengchen/glassdoor/data/processed/review_cons_dic_mat.pkl', 'rb')
[dictionary_cons, doc_term_matrix_cons] = pickle.load(file)
file.close()
#%% assign topics to the reviews (pros)
topics_words = lda_model.get_topics() # 9*3704
X = gensim.matutils.corpus2dense(doc_term_matrix, len(dictionary)) # converse the sparse matrix to normal dense matrix
X = X.T
print(X.shape) # 163720 * 3704   (# reviews * # topics)
#%%
ntopics = 9
doc_topics = np.zeros((np.shape(X)[0], ntopics))
for i in range(np.shape(X)[0]):
    if i%5000 == 0:
        print(i)
    topic_tuples = lda_model.get_document_topics(doc_term_matrix[i], 0, 0, True)[0]
    topic_scores = np.zeros((1, len(topic_tuples)))
    for j, score in enumerate(topic_tuples):
        topic_scores[0,j] = score[1]
    doc_topics[i,:] = topic_scores
file = open('/Users/chengchen/glassdoor/data/processed/lda_9_pros_scores.pkl', 'wb')
pickle.dump([doc_topics, topics_words], file)
file.close()


topics_words_cons = lda_model_cons.get_topics() # 9*3704
X_cons = gensim.matutils.corpus2dense(doc_term_matrix_cons, len(dictionary_cons)) # converse the sparse matrix to normal dense matrix
X_cons = X_cons.T
ntopics = 8
doc_topics_cons = np.zeros((np.shape(X_cons)[0], ntopics))
for i in range(np.shape(X_cons)[0]):
    if i%5000 == 0:
        print(i)
    topic_tuples = lda_model_cons.get_document_topics(doc_term_matrix_cons[i], 0, 0, True)[0]
    topic_scores = np.zeros((1, len(topic_tuples)))
    for j, score in enumerate(topic_tuples):
        topic_scores[0,j] = score[1]
    doc_topics_cons[i,:] = topic_scores
file = open('/Users/chengchen/glassdoor/data/processed/lda_8_cons_scores.pkl', 'wb')
pickle.dump([doc_topics_cons, topics_words_cons], file)
file.close()

#%% manually add topic by eyeballing the topic keyword composition
topics_pros = ['career opportunities',
               'management and supportive work environment',
               'salary and benefits',
               'flexible working schedule',
               'friendly to junior co-workers',
               'culture and value',
               'friendly environment with nice and smart collegues',
               'work-life balance',
               'food, discount, and other perks'
               ]
df_topics_pros = pd.DataFrame(doc_topics, columns = topics_pros)

#%% add topics of reviews on cons
topics_cons = ['slow to change and adapt',
               'limited opportunity for career growth',
               'demanding work',
               'pressure from sales and customer service',
               'bad managers and employees',
               'poor management and communication',
               'low payment and high turnover ratio',
               'long working hours'
               ]
df_topics_cons = pd.DataFrame(doc_topics_cons, columns = topics_cons)
                              
#%% load the original data and add the identified topics
file = open('/Users/chengchen/glassdoor/data/processed/review_subset.pkl', 'rb')
df = pickle.load(file)
file.close()
df = df.drop_duplicates(subset =['review_pros', 'review_cons'])
df.shape # 163720*17
df_topics_pros.index = df.index
df_topics_cons.index = df.index
df_review_topics = pd.concat([df, df_topics_pros, df_topics_cons], axis = 1)
#%% merge with company profile
file = open('/Users/chengchen/glassdoor/data/processed/company_rating.pkl', 'rb')
df_company = pickle.load(file)
file.close()
df_all = pd.merge(df_review_topics, df_company, left_on = 'company_name', right_on = 'company_name', how = 'left')
df_all.to_csv('/Users/chengchen/glassdoor/data/processed/processed_reviews.csv', index = False)
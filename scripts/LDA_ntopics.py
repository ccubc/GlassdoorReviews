#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:15:54 2019

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

file = open('/Users/chengchen/glassdoor/data/processed/lemma_reviews.pkl', 'rb')
[review_pros,review_cons] = pickle.load(file)
file.close()

def create_dic_and_matrix(reviews):
    """ creating dictionary and bag of words from review text for LDA model"""
    dictionary = corpora.Dictionary(reviews)
    # filter out tokens that are too common or too rare
    dictionary.filter_extremes(no_above = 0.25, no_below = 10)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews]
    return dictionary, doc_term_matrix

def LDA_with_varying_ntopics(ntopics_list, reviews, file_name):
    """ train LDA with a range of n_topics(number of topics) and save the coherence score"""
    processed_info = copy.deepcopy(reviews)
    dictionary, doc_term_matrix = create_dic_and_matrix(reviews)
    scores = []
    myseed = 0
    for ntopics in ntopics_list:
        print(ntopics)
        lda_model =  gensim.models.LdaMulticore(doc_term_matrix, num_topics = ntopics, id2word = dictionary, 
                                        random_state=myseed, passes = 10, workers = 2) #iterations = 300
        coherence_model_lda = CoherenceModel(model=lda_model, texts = processed_info, corpus=doc_term_matrix, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        scores.append(coherence_lda)
    file = open(file_name, 'wb')
    pickle.dump(scores, file)
    file.close()

ntopics_list = np.arange(5,30)
file_name = '/Users/chengchen/glassdoor/data/processed/scores_coherence_cv_5_30.pkl'
LDA_with_varying_ntopics(ntopics_list, review_pros, file_name)
file_name2 = '/Users/chengchen/glassdoor/data/processed/scores_coherence_cv_cons_5_30.pkl'
LDA_with_varying_ntopics(ntopics_list, review_cons, file_name2)

# plot the coherence scores against the number of topics
def plot_CV_ntopics(ntopic_list, score_file_name, plot_file_name):
    file = open(score_file_name, 'rb')
    scores = pickle.load(file)
    file.close()
    sns.set('paper')
    plt.plot(ntopic_list, scores)
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence Score (CV)')
    plt.savefig(plot_file_name)
plot_CV_ntopics(ntopics_list, file_name, '/Users/chengchen/glassdoor/graphs/review_pro_coherence_score_5_30.png')
plot_CV_ntopics(ntopics_list, file_name2, '/Users/chengchen/glassdoor/graphs/review_cons_coherence_score_5_30.png')

# Glassdoor Review

This is an ongoing project. <br>
scripts and notebooks:
* script: read_large_dta.py: reads the original 16GB STATA data file, and randomly select a representative subsample to conduct text analysis
* script: data_preprocessing.py: pre-process reviews on company pros and cons by: removing stop words + lemmatization
* script: LDA_ntopics.py: trying LDA model with different number of topics and plot the coherence scores to find optimal number of topics to set for LDA model

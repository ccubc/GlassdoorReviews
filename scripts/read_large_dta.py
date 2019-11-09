#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:48:31 2019

@author: chengchen
"""

import pandas as pd
import time
import seaborn as sns
import datetime as dt
import pickle

pd.set_option('mode.chained_assignment', None)
#%%
data_path = '/Users/chengchen/glassdoor/data/glassdoor.dta'

chunksize = 500000 # data is too large, read by chunks of 500,000
chunk_count = 0
df_company = pd.DataFrame(columns = ['review_year'])
for df in pd.read_stata(data_path, chunksize = chunksize, iterator = True, 
                        columns = ['company_name','industry','review_date']):
    chunk_count += 1
    iter_start = dt.datetime.now()
    print('iteration {} starts'.format(chunk_count))
    print(' {} rows'.format(chunk_count*chunksize))
    df['review_year'] = pd.DatetimeIndex(df['review_date']).year
    df_company_batch = pd.DataFrame(df.groupby(['company_name','review_year'])['review_year'].count())
    df_company = pd.concat([df_company, df_company_batch])
    print('iteration {} completed in {} seconds'.format(chunk_count,(dt.datetime.now() - iter_start).seconds))
#%%
# unstack the index of company-year

df_company = df_company.reset_index()  
df_company[['company', 'year']] = pd.DataFrame(df_company['index'].tolist(), index=df_company.index) 

#%% histogram of number of reviews by year
df_year = pd.DataFrame(df_company.groupby(['year'])['review_year'].sum())
x = df_year.index
y = df_year.review_year
sns.barplot(x = x, y = y, palette = "deep")

# year 2016 has the most reviews. Will create a subsample of 2016 and just analyze that.

#%% next step is to check the distribution of reviews per company
df_company2 = pd.DataFrame(df_company[df_company['year']==2016].groupby(['company'])['review_year'].sum())
df_company2.describe()
# there are 179386 companies that have reviews in year 2016
df_company2.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')

#%%
df_company3 = df_company2[df_company2['review_year']>4][df_company2['review_year']<100]
df_company3.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
#%%
# 179386 companies have reviews in year 2016
# 37670 companies have >5 reviews in year 2016
# 21460 companies have >10 reviews in year 2016
df_company2[df_company2['review_year']>20].shape
# 11334 companies have >20 reviews in year 2016
# will just use these 11334 companies
company_subsample = df_company2[df_company2['review_year']>20].index.tolist()
# a list of company names that I will use
#%% save this company list
# save preprocessed data
file = open('/Users/chengchen/glassdoor/data/processed/company_list.pkl', 'wb')
pickle.dump([company_subsample], file)
file.close()
#%% load saved company list

file = open('/Users/chengchen/glassdoor/data/processed/company_list.pkl', 'rb')
company_list = pickle.load(file)[0]
file.close()
#%%
company_columns = ['company_name',
       'number_of_employees', 
       'company_type', 'industry', 
       'number_of_reviews', 
       'company_overal_rating',
       'company_culture_and_value_rating', 
       'company_work_life_balance_rating',
       'company_senior_management_rating', 
       'company_comp_and_benefits_rating',
       'compary_career_opportunities_rat','review_date']

chunk_count = 0
for df in pd.read_stata(data_path, chunksize = 500000, columns = company_columns, iterator = True):
    chunk_count += 1
    iter_start = dt.datetime.now()
    print('iteration {} starts'.format(chunk_count))
    df['review_year'] = pd.DatetimeIndex(df['review_date']).year
    df_subsample = df[df['review_year']==2016]
    df_subsample = df_subsample[df_subsample['company_name'].isin(company_list)]
    df_subsample.drop_duplicates(subset = 'company_name', inplace = True)
    if chunk_count == 1:
        df_sub = df_subsample
    else:
        df_sub = pd.concat([df_sub, df_subsample])
    print('sub sample has {} rows in total'.format(df_sub.shape[0]))
    print('iteration {} completed in {} seconds'.format(chunk_count,(dt.datetime.now() - iter_start).seconds))
df_sub.drop_duplicates(subset = 'company_name', inplace = True)
file = open('/Users/chengchen/glassdoor/data/processed/company_rating.pkl', 'wb')
pickle.dump(df_sub, file)
file.close()
#%%
review_columns = [ 'company_name',
       'review_date',
       'review_overall_rating', 
       'review_culture_and_value_rating',
       'review_work_life_balance_rating', 
       'review_senior_management_rating',
       'review_comp_and_benefits_rating', 
       'review_career_opportunities_rati',
       'number_of_people_found_this_revi',
       'review_job_title', 
       'review_employee_status',
       'reviewer_location', 
       'reviewer_form_of_employment',
       'reviewer_lenght_of_employment', 
       'review_pros', 
       'review_cons']
chunk_count = 0
for df in pd.read_stata(data_path, chunksize = 100000, columns = review_columns, iterator = True):
    chunk_count += 1
    iter_start = dt.datetime.now()
    print('iteration {} starts'.format(chunk_count))
    df['review_year'] = pd.DatetimeIndex(df['review_date']).year
    df_subsample = df[df['review_year']==2016]
    df_subsample = df_subsample[df_subsample['company_name'].isin(company_list)]
    df_subsample = df_subsample.sample(frac=0.1, random_state=1)
    if chunk_count == 1:
        df_sub = df_subsample
    else:
        df_sub = pd.concat([df_sub, df_subsample])
    print('sub sample has {} rows in total'.format(df_sub.shape[0]))
    print('iteration {} completed in {} seconds'.format(chunk_count,(dt.datetime.now() - iter_start).seconds))
file = open('/Users/chengchen/glassdoor/data/processed/review_subset.pkl', 'wb')
pickle.dump(df_sub, file)
file.close()

#%%












                                                                                                                      
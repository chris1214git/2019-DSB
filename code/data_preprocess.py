#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from IPython.display import display


from utils import kaggle_util


# In[2]:


# https://www.kaggle.com/robikscube/2019-data-science-bowl-an-introduction

import seaborn as sns
sns.set_style("whitegrid")
my_pal = sns.color_palette(n_colors=10)

## https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_option.html#pandas.get_option
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)


# In[3]:


get_ipython().run_cell_magic('time', '', '\n# read data\ntrain, test, train_labels, specs, sample_submission = kaggle_util.read_data()\ntrain2, test2, train_labels2 = train.copy(), test.copy(), train_labels.copy()\n\n# get usefull dict with maping encode, we implement "label encoding" on category features\ntrain2, test2, train_labels2, win_code, list_of_user_activities, list_of_event_code,\\\nactivities_labels, assess_titles, list_of_event_id, all_title_event_code = kaggle_util.encode_title(train2, test2, train_labels2)\n\n# tranform function to get the train and test set\nreduce_train, reduce_test, categoricals = kaggle_util.get_train_and_test(train2, test2, win_code, list_of_user_activities, list_of_event_code,\\\nactivities_labels, assess_titles, list_of_event_id, all_title_event_code)')


# In[4]:


reduce_train


# In[5]:


# reduce_train.to_csv('../data/preprocess/'+'train895.csv', index=False)
# reduce_test.to_csv('../data/preprocess/'+'test895.csv', index=False)
reduce_train.to_csv('../data/preprocess/'+'train_1225_2.csv', index=False)
reduce_test.to_csv('../data/preprocess/'+'test_1225_2.csv', index=False)


# In[6]:


reduce_train.isnull().sum().sum()


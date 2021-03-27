#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def accuracy_score(y_true, y_predict, percent=None):
    if percent==None:
        y_predict = y_predict>=0.5
    else:
        limit = round(y_true.shape[0]/100*percent)
        y_true = y_true[:limit]
        y_predict = y_predict[:limit]
    result = np.sum(y_true==y_predict, axis = 0)/y_true.shape[0]
    return np.mean(result)


# In[3]:


def precision_score(y_true, y_predict, percent=None):
    if percent==None:
        y_predict = y_predict>=0.5
    else:
        limit = round(y_true.shape[0]/100*percent)
        y_true = y_true[:limit]
        y_predict = y_predict[:limit]
    tp = 0
    if len(y_true.shape) == 1:
        for i in range(y_true.shape[0]):
                if (y_true[i]==y_predict[i]==1):
                    tp+=1
    else: #если y_predict имеет несколько классов
        for j in range(y_true.shape[1]):
            tp = 0
            for i in range(y_true.shape[0]):
                if (y_true[i][j]==y_predict[i][j]) and (y_predict[i][j]==1):
                    tp+=1
    result = tp/np.sum(y_predict, axis = 0)
    return np.mean(result)


# In[4]:


def recall_score(y_true, y_predict, percent=None):
    if percent==None:
        y_predict = y_predict>=0.5
    else:
        limit = round(y_true.shape[0]/100*percent)
        y_true = y_true[:limit]
        y_predict = y_predict[:limit]
    tp=0
    if len(y_true.shape) == 1:
        for i in range(y_true.shape[0]):
                if (y_true[i]==y_predict[i]==1):
                    tp+=1
    else: 
        for j in range(y_true.shape[1]):
            tp = 0
            for i in range(y_true.shape[0]):
                if (y_true[i][j]==y_predict[i][j]) and (y_predict[i][j]==1):
                    tp+=1
    result = tp/np.sum(y_true, axis = 0)
    return np.mean(result)


# In[5]:


def lift_score(y_true, y_predict, percent=None):
    if percent==None:
        y_predict = y_predict>=0.5
    else:
        limit = round(y_true.shape[0]/100*percent)
        y_true = y_true[:limit]
        y_predict = y_predict[:limit]
    precision = precision_score(y_true, y_predict, percent)
    result = precision/(np.sum(y_true, axis = 0)/y_true.shape[0])
    return np.mean(result)


# In[6]:


def f1_score (y_true, y_predict, percent=None):
    if percent==None:
        y_predict = y_predict>=0.5
    else:
        limit = round(y_true.shape[0]/100*percent)
        y_true = y_true[:limit]
        y_predict = y_predict[:limit]
    precision = precision_score(y_true, y_predict, percent)
    recall = recall_score(y_true, y_predict, percent)
    result = 2*precision*recall/(precision+recall)
    return result


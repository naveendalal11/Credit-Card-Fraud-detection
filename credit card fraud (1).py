#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import seaborn as sn


# In[3]:


fraud=pd.read_csv('https://www.dropbox.com/s/6qcgvoc6h8y8zb2/CreditCardDefault.csv?dl=1')


# In[5]:


fraud.head()


# In[7]:


print('The dataset contains {0} rows and {1} columns.'.format(fraud.shape[0], fraud.shape[1]))


# In[9]:


fraud.info()


# In[4]:


import missingno as msno


# In[5]:


msno.bar(fraud)


# In[6]:


msno.matrix(fraud)


# In[10]:


print('Normal transactions count: ', fraud['Class'].value_counts().values[0])
print('Fraudulent transactions count: ', fraud['Class'].value_counts().values[1])


# In[11]:


## DEfine x and y where x is dependent and y is independent variable
X = fraud.iloc[:, :-1]


# In[12]:


y = fraud['Class']


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


## Standardize data
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


# In[18]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[20]:


## Partition data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=10000)


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[23]:


## Train models
LR=LogisticRegression()


# In[24]:


LR.fit(X_train,y_train)


# In[25]:


y_pred=LR.predict(X_train)


# In[26]:


y_pred


# In[27]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[28]:


accuracy_score(y_train,y_pred)


# In[29]:


confusion_matrix(y_train,y_pred)


# In[30]:


print(classification_report(y_train,y_pred))


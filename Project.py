#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('/Users/serenayang/Desktop/DSC478/project/dataset/train.csv', sep = ',', header = 0)


# In[3]:


df


# In[4]:


#search all and output any row with nulls
df[df.isnull().any(axis = 1)]


# In[5]:


#drop all NA in employement_type 
df = df.dropna(subset = ['EMPLOYMENT_TYPE'])


# In[6]:


import datetime as dt
pd.options.mode.chained_assignment = None  # default='warn'

#format dataset into correct type
df['DATE_OF_BIRTH'] =  pd.to_datetime(df['DATE_OF_BIRTH'], format='%d-%m-%Y')
df['DISBURSAL_DATE'] =  pd.to_datetime(df['DISBURSAL_DATE'], format='%d-%m-%Y')
df['DATE_OF_BIRTH'] = df['DATE_OF_BIRTH'].map(dt.datetime.toordinal)
df['DISBURSAL_DATE'] = df['DISBURSAL_DATE'].map(dt.datetime.toordinal)


# In[7]:


#split AVERAGE_ACCT_AGE and CREDIT_HISTORY_LENGTH into year month
df[['AVERAGE_ACCT_YR', 'AVERAGE_ACCT_M']] = df['AVERAGE_ACCT_AGE'].str.split(' ', expand = True)
df['AVERAGE_ACCT_YR'] = df['AVERAGE_ACCT_YR'].str.extract('(\d+)')
df['AVERAGE_ACCT_M'] = df['AVERAGE_ACCT_M'].str.extract('(\d+)')
#to numeric
df['AVERAGE_ACCT_YR'] = pd.to_numeric(df['AVERAGE_ACCT_YR'])
df['AVERAGE_ACCT_M'] = pd.to_numeric(df['AVERAGE_ACCT_M'])

df[['CREDIT_HISTORY_LENGTH_YR', 'CREDIT_HISTORY_LENGTH_M']] = df['CREDIT_HISTORY_LENGTH'].str.split(' ', expand = True)
df['CREDIT_HISTORY_LENGTH_YR'] = df['CREDIT_HISTORY_LENGTH_YR'].str.extract('(\d+)')
df['CREDIT_HISTORY_LENGTH_M'] = df['CREDIT_HISTORY_LENGTH_M'].str.extract('(\d+)')
#to numeric
df['CREDIT_HISTORY_LENGTH_YR'] = pd.to_numeric(df['CREDIT_HISTORY_LENGTH_YR'])
df['CREDIT_HISTORY_LENGTH_M'] = pd.to_numeric(df['CREDIT_HISTORY_LENGTH_M'])

#drop the old AVERAGE_ACCT_AGE and CREDIT_HISTORY_LENGTH
df = df.drop(columns = ['AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH'])


# In[8]:


df.info()


# In[9]:


#turn categorical attributes into dummy variables
df = pd.get_dummies(df, columns = ['EMPLOYMENT_TYPE', 'PERFORM_CNS_SCORE_DESCRIPTION'])

#drop unique ID
df = df.drop(columns = ['UNIQUEID'])


# In[10]:


classStats = df.groupby('LOAN_DEFAULT').size()
classStats.plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Loan Default')
plt.yticks(np.arange(0, max(classStats), 25000))
plt.show()


# ----

# ## KNN

# In[11]:


target = df.LOAN_DEFAULT
df = df.drop(columns = ['LOAN_DEFAULT'])
target.shape


# In[12]:


target.head()


# In[13]:


#seprate into test and train sets
from sklearn.model_selection import train_test_split

train, test, train_label, test_label = train_test_split(df, target, test_size=0.2, random_state=33)
print(train.shape, test.shape)


# In[14]:


print(train_label.shape, test_label.shape)


# In[15]:


train.head()


# In[16]:


#Run the accuracy function on a range of values for K in order to compare accuracy values for different 
#numbers of neighbors. Do this both using Euclidean Distance as well as Cosine similarity measure. 
#For example, when we try evaluating the classifiers on a range of values of K from 1 through 20 and 
#present the results as a table or a graph.

trainA = np.array(train)
testA = np.array(test)
traintblF = np.array(train_label).flatten()
testtblF = np.array(test_label).flatten()


# In[18]:


#Naive Bayes
from sklearn import neighbors, tree, naive_bayes
nbclf = naive_bayes.GaussianNB()
nbclf = nbclf.fit(train, train_label)
nbpreds_test = nbclf.predict(test)
print (nbpreds_test)


# In[19]:


## compute the average accuracy score across the test instances
print (nbclf.score(train, train_label))


# In[20]:


#compared to the performance on the training data itself (to check for over- or under-fitting)
print (nbclf.score(test, test_label))


# In[26]:


from sklearn.metrics import confusion_matrix
nbccm = confusion_matrix(test_label, nbpreds_test, labels=[0,1])
print(nbccm)


# In[27]:


plt.matshow(nbccm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[30]:


from sklearn.model_selection import cross_val_score
nbcv_scores = cross_val_score(nbclf, df, target, cv=10)
print(nbcv_scores)


# In[31]:


print("Overall Average Accuracy for Naive Bayes (Gaussian): %0.2f (+/- %0.2f)" % (nbcv_scores.mean(), nbcv_scores.std() * 2))


# In[32]:


#scikit-learn's KNN classifier
#Performing min-max normalization to rescale numeric attributes.
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler().fit(train)
data_trainN = min_max_scaler.transform(train)
data_testN = min_max_scaler.transform(test)


# In[ ]:


from sklearn import neighbors, tree, naive_bayes
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

test = []
train = []
avgResult = []
result = {}
for i in range(1,21):

    knnclf = neighbors.KNeighborsClassifier(i, weights='distance')
    knnclf.fit(data_trainN, train_label)
    #Next, we call the predict function on the test intances to produce the predicted classes.
    knnpreds_test = knnclf.predict(data_testN)
    
    #scikit-learn has various modules that can be used to evaluate classifier accuracy
    print(classification_report(test_label, knnpreds_test))
    
    #confusion matrix
    knncm = confusion_matrix(test_label, knnpreds_test)
    print(knncm)
    
    #get percentahe score from test and train dataset
    # compute the average accuracy score across the test instances
    testScore = knnclf.score(data_testN, test_label)
    #compared to the performance on the training data itself (to check for over- or under-fitting)
    trainScore = knnclf.score(data_trainN, train_label)
    avgScore = (trainScore + testScore)/2
    
    train.append(trainScore.round(3))
    test.append(testScore.round(3))
    avgResult.append(avgScore.round(3)) 
    


# In[ ]:


result['train'] = train
result['test'] = test
result['avg'] = avgResult
withWeighting = pd.DataFrame(results, columns=['train', 'test', 'avg'])
withWeighting


# In[ ]:





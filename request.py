#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


# In[2]:



df = pd.read_json('News_Category_Dataset_v2.json', lines=True)


# In[3]:


#df.head()


# In[4]:





# In[5]:


df.dropna(inplace = True)


# In[6]:


#df.shape


# In[7]:


#df.head()


# In[17]:

cp = df['headline'] 
df["Text"] = cp

cp = df["category"]
df["Cat1"] = cp

df1 = df[["Text","Cat1"]]


# In[18]:


#df1.shape


# In[19]:


enc = LabelEncoder().fit(df1.Cat1)
encoded = enc.transform(df1.Cat1)


# In[20]:


#df.head()


# In[128]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Text, encoded,test_size=0.03, random_state=42)


# In[94]:
print(" -------- -- -- -- -- -",end='')

hh = sys.argv[1]
hh = str(hh)
print("Input text = ", hh," ||  ", end="    ")

# In[129]:


X_test.reset_index(drop=True,inplace=True)


# In[130]:


X_test[0] = hh


# In[131]:


#X_test


# In[132]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", decode_error="ignore")
vectorizer.fit(X_train)


# In[133]:


X_train_vectorized = vectorizer.transform(X_train)


# In[134]:


from sklearn.naive_bayes import MultinomialNB
cls = MultinomialNB()
# transform the list of text to tf-idf before passing it to the model
cls.fit(vectorizer.transform(X_train), y_train)
 


# In[135]:


from sklearn.metrics import classification_report, accuracy_score
#hh = str(input("Enter the text "))
y_pred = cls.predict(vectorizer.transform(X_test))


# In[ ]:





# In[ ]:





# In[136]:


lss = enc.inverse_transform(y_pred)


# In[137]:


print("The category matched is =",end="")
print(lss[0],end = " ")



# In[ ]:





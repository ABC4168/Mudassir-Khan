#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df.shape


# There are 8124 rows and 23 columns are present in the Mushroom Report.

# In[6]:


df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/mushrooms.csv")
df.head()


# In[10]:


df.tail()


# In[14]:


df.info()


# In[8]:


df.isnull().sum()


# There are no null values present in the given report.

# In[15]:


sns.countplot(df['class'])


# In[16]:


plt.figure(figsize=(12,6))
h=sns.catplot(x="cap-shape",hue="cap-color",col="class",
             data=df, kind="count",height=7, aspect=.6);


# 1:White Coloured and Bell Shaped Mushrooms are highly recommended for eating
# 2:Red Coloured Knobbed Shaped Mushrooms are poisonous

# In[18]:


plt.figure(figsize=(12,6))
h=sns.catplot(x="odor",hue="bruises",col="class",
             data=df, kind="count",height=7, aspect=.6,palette='inferno');


# 1:Odourless and bruised Mushrooms are highly recommended for eating
# 2:Fishy Odour with No Bruises Mushrooms are poisonous

# In[21]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df=df.apply(LabelEncoder().fit_transform)
df.head()


# In[22]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df=df.apply(LabelEncoder().fit_transform)
df.tail()


# In[23]:


x=df.drop(['class'],axis=1)
y=df['class']


# In[24]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state = 0)


# Decision Tree Model

# In[26]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state = 0 ,max_depth = 4)
dt.fit(x_train , y_train)


# In[27]:


dt.score(x_train , y_train)


# In[31]:


predictions = dt.predict(x_test)


# In[32]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test , predictions)


# Random Forest Model

# In[33]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 4)


# In[34]:


rf.fit(x_train , y_train)
rf.score(x_train , y_train)


# In[36]:


predictions = rf.predict(x_test)
rf.score(x_test , y_test)


# In[37]:


accuracy_score(y_test , predictions)


# According to the survey:
#     1:Decision Tree Accuracy-98.2%
#     2:random Forest accuracy-99.1%

# In[ ]:





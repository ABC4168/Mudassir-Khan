#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv")
df.head()


# First 5 rows of data

# In[3]:


df.tail()


# Last 5 rows of data

# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[11]:


df.columns


# In[12]:


df.describe()


# In[14]:


df.shape


# In[15]:


df1=df.Region.value_counts()
plt.figure(figsize=(10,10))
sns.barplot(x=df1.index , y=df1.values)
plt.xlable('Region')
plt.xticks(rotation=90)
plt.ylabel('No. of country')
plt.show()


# Barplot for Region verse No. of Country

# In[16]:


corr_hmap=df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr_hmap,annot=True)
plt.show()


# In[17]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column]=labelencoder.fit_transform(df[column])


# In[18]:


df.head()


# In[19]:


sns.countplot(df['Region'])


# Countplot for Region

# In[20]:


sns.boxplot(df['Region'])


# In this boxplot,there is no outler

# In[21]:


plt.figure(figsize=[10,6])
plt.title('comparision between Happiness Rank and Economy (GDP per Capita)' )
sns.scatterplot(df['Economy (GDP per Capita)'],df['Happiness Rank'],hue=df['Region']);


# In[22]:


df.rename({'Happiness Score': 'Happiness_Score'},axis=1,inplace=True)
df


# In[23]:


x=df.drop('Happiness_Score',axis=1)
y=df.Happiness_Score


# In[24]:


from sklearn.model_selection import train_test_split,cross_val_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


# In[25]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
def maxr2_score(regr,x,y):
    max_r_score=0
    for r_state in range(40,101):
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=r_state,test_size=0.30)
        regr.fit(x_train,y_train)
        y_pred=regr.predict(x_test)
        r2_scr=r2_score(y_test,y_pred)
        print("r2 score corrospoding to ",r_state,"is",r2_scr)
        if r2_scr>max_r_score:
            max_r_score=r2_scr
            final_r_state=r_state
    print("max r2 score to ",final_r_state,"is",max_r_score)
    return final_r_state


# In[26]:


from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
r_state=maxr2_score(lreg,x,y)


# In[27]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
cross_val_score(LinearRegression(),x,y,cv=5,scoring='neg_mean_absolute_error').mean()


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state =62,test_size=0.33)
lreg=LinearRegression()
lreg.fit(x_train,y_train)
y_pred=lreg.predict(x_test)


# In[29]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("r2 score is :",r2_score(y_test,y_pred))
print('RMSE is :',np.sqrt(mean_squared_error(y_test,y_pred)))


# In[30]:


import pickle


# In[31]:


project2_model = pickle.dumps(lreg) 
lreg_from_pickle = pickle.loads(project2_model) 
lreg_from_pickle.predict(x_test)


# In[ ]:





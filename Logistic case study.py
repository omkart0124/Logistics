#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing some necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


d= pd.read_csv(r"D:\ML\train.csv")
# using pd i redd my csv file


# In[3]:


d


# In[4]:


d.shape


# In[ ]:





# # cleaning data

# In[5]:


d.isnull()


# In[6]:


d.isnull().sum()


# In[7]:


sns.heatmap(d.isnull(), yticklabels=False,cbar=False, cmap='viridis')#cbar=False, cmap='viridis'
plt.show()
# this is a heatmap concept its help us to see that whatever condition we put over here and whichever is true
# that will be display in another color
# here i want to show some parameters such as 'yticklabels' as false so I've 2axis x and y
# in this heatmap all null values shown in yellow color
# cmap for color
# yellow stands for null
# we have missing values in age, cabin and Embarded
# here we can understand which colom have more nan values


# In[ ]:





# In[8]:


sns.set_style('darkgrid')
sns.countplot(x='Survived',data=d)
plt.show()

# survived =  0 means person didnot survied and  1 means survived
# based on survived column iam going to plot a counter plot
# count plot help me to see that what is the count of survived and not survived
# more that 500 people hasn't  survived and over 300 survived


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue= 'Sex',data=d,palette='RdBu_r')
plt.show()

# using counterpot on x is survived hue is based on survived
# palette is just for good color graphs
# on x ive 0 and 1
# people who didnot survied is mostly male and who survived is mostly female


# In[ ]:





# In[ ]:





# In[10]:


d['Age'].replace("NaN", np.nan, inplace=True)


# In[11]:


d['Age']=d['Age'].astype(float)


# In[12]:


d['Age'].dtype


# In[13]:


sns.distplot(d['Age'])
plt.axvline(d['Age'].mean(), color="red")
# use of axvline when you want to draw straight line on the x-axis(we want to draw mean value)
plt.axvline(d['Age'].median(), color="yellow")
plt.show()


# In[14]:


# Not much diff in them


# In[15]:


d['Age'].mean()


# In[16]:


d['Age'].median()# lower than mean


# In[17]:


d['Age'].fillna(d['Age'].median(), inplace=True)


# In[18]:


sns.heatmap(d.isnull(), yticklabels=False,cbar=False, cmap='viridis')
plt.show()


# In[ ]:





# In[19]:


#droping cabing column values

d.drop('Cabin', axis=1, inplace=True)


# In[20]:


d.head(2)

#sibsp is basically the count of spouse and space
# Parch this is the total no of parents and children


# In[21]:


d["Embarked"]=d["Embarked"].fillna("Unknown")


# In[22]:


# d.info()


# In[ ]:





# In[23]:


sns.heatmap(d.isnull(), yticklabels=False,cbar=False, cmap='viridis')
plt.show()

# here we can see there is no nan values


# In[ ]:





# In[24]:


d.head()


# In[25]:


# here i converted sex and embarked into dummies variables


# In[26]:


pd.get_dummies(d['Embarked'],drop_first=True).head()


# In[27]:


sex=pd.get_dummies(d['Sex'],drop_first=True)
embark=pd.get_dummies(d['Embarked'],drop_first=True)


# In[ ]:





# In[28]:


d.drop(['PassengerId','Sex','Embarked','Name','Ticket'], axis=1, inplace=True)


# In[29]:


d.head()


# In[30]:


# here i concate all my new colomns into the data set


# In[31]:


d=pd.concat([d,sex,Embarked],axis=1)


# In[ ]:


d.head()


# In[ ]:


#d.drop('Unknown',axis=1, inplace=True)


# In[ ]:


#d.head()
# this is my final dataset


# In[ ]:


# Survived column is my dependend variable and remaining is independent


# # train test split

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


# here i define independent and dependent variable  

x = d.drop('Survived',axis = 1)
y = d['Survived']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)


# In[ ]:


log = LogisticRegression()    # object created
log.fit(x_train,y_train)      # fit train data of x and y


# In[ ]:


predict = log.predict(x_test)   # for x_test data i get my prediction


# In[ ]:


from sklearn.metrics import confusion_matrix          # import confusion matrix


# In[ ]:


accuracy = confusion_matrix(y_test,predict)


# In[ ]:


accuracy


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.metrics import (accuracy_score)


# In[ ]:


accuracy_score(y_test,predict)
# accuracy is 77 which is good
# (130+77)207/268(all no)


# In[ ]:


predict


# In[ ]:





# In[ ]:





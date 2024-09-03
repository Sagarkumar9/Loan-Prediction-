#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import seaborn as snsb


# In[2]:


df = pd.read_csv("C:/Users/USER/Desktop/loan.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


#we'll get full infomation on our data, like count, Datatype.(eg- int,float,string,object...)
df.info()


# In[6]:


#we'll know how many missing values are present in our dataset
df.isnull().sum()


# In[7]:


#creating new column LoanAmount_log. This is commonly done to reduce skewness and normalize the data.
df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[8]:


df['TotalIncome']= df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)


# In[9]:


#filling NA values, mean when it's continuous values and mode when it's categorical value

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log=df.LoanAmount_log.fillna(df.LoanAmount_log.mean())

df.isnull().sum()


# In[10]:


#we'll select specific columns for training and testing
#iloc is used for positional indexing, and np.r_[] helps combine the column ranges.
#The ".values" converts the selected DataFrame columns into numpy arrays

x=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y=df.iloc[:,12].values

x


# In[11]:


y


# In[12]:


#purpose of this code is to assess the quality of the data in the Gender column by determining how much data is missing.

print("percent of missing gener is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[13]:


print("number of people who take loan as group by gender")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df, palette='Set2')


# In[14]:


print('people who take loan as meretail status')
print(df['Married'].value_counts())
sns.countplot(x="Married",data=df, palette='Set1')


# In[15]:


print('number of people who take loan as dependent')
print(df["Dependents"].value_counts())
sns.countplot(x='Dependents',data=df,palette='Set1')


# In[16]:


print("people who take loan as self employeed")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=df,palette='Set1')


# In[22]:


print("people who take loan as Loan amount")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=df,palette='Set1')


# In[19]:


print("people who take loan from credit history")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=df,palette='Set1')


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Splitting the data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

# Instantiate the LabelEncoder
Labelencoder_x = LabelEncoder()

# Loop through the specified columns to encode categorical variables
for i in range(0, 5):
    xtrain[:, i] = Labelencoder_x.fit_transform(xtrain[:, i])
xtrain[:, 7] = Labelencoder_x.fit_transform(xtrain[:, 7])

xtrain


# In[33]:


Labelencoder_y = LabelEncoder()
ytrain = Labelencoder_y.fit_transform(ytrain)

ytrain


# In[34]:


for i in range(0,5):
    xtest[:,i]= Labelencoder_x.fit_transform(xtest[:,i])
    xtest[:,7]= Labelencoder_x.fit_transform(xtest[:,7])

xtest


# In[35]:


Labelencoder_y = LabelEncoder()
ytest = Labelencoder_y.fit_transform(ytest)

ytest


# In[40]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xtrain= ss.fit_transform(xtrain)
xtest= ss.fit_transform(xtest)


# In[41]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(xtrain,ytrain)


# In[43]:


from sklearn import metrics
y_pred = rfc.predict(xtest)

print("accurecy=",metrics.accuracy_score(y_pred, ytest))

y_pred 


# In[ ]:


#in the above output '1' denotes loan to be approved and '0' denotes loan to be rejected


# In[47]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(xtrain,ytrain)

ypred=dtc.predict(xtest)
print('accuracy=',metrics.accuracy_score(y_pred,ytest))


# In[48]:


y_pred


# In[49]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(xtrain,ytrain)

y_pred=nb.predict(xtest)
print("accuracy=",metrics.accuracy_score(y_pred,ytest))


# In[50]:


y_pred


# In[ ]:


#best model is GaussianNB as it gave an accuracy of 82.92%


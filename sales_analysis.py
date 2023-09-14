#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


add=pd.read_csv("Advertising.csv")


# In[4]:


add


# In[5]:


add.head()


# In[6]:


add.tail()


# In[7]:


add.describe()


# In[8]:


add.info()


# In[9]:


add.shape


# In[10]:


sns.heatmap(add.corr(),annot=True)


# In[11]:


imp_feature=list(add.corr()['Sales'][(add.corr()['Sales']>+0.5)|(add.corr()['Sales']<-0.5)].index)


# In[12]:


print(imp_feature)


# In[13]:


x=add['TV']
y=add['Sales']


# In[14]:


x=x.values.reshape(-1,1)


# In[15]:


x


# In[16]:


y


# In[17]:


print(x.shape,y.shape)


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)


# In[19]:


x_train


# In[20]:


x_test


# In[21]:


y_train


# In[22]:


y_test


# In[23]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[24]:


Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)
x_test_scaled=Sc.fit_transform(x_test)


# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


lr=LinearRegression()


# In[27]:


lr.fit(x_train_scaled,y_train)


# In[28]:


y_pred=lr.predict(x_test_scaled)


# In[29]:


from sklearn.metrics import r2_score


# In[30]:


r2_score(y_test,y_pred)


# In[31]:


plt.scatter(y_test,y_pred,c='b')


# In[32]:


from sklearn.neighbors import KNeighborsRegressor


# In[33]:


knn=KNeighborsRegressor().fit(x_train,y_train)


# In[34]:


knn


# In[35]:


knn_train_pred=knn.predict(x_train)


# In[36]:


knn_test_pred=knn.predict(x_test)


# In[37]:


print(knn_train_pred,knn_test_pred)


# In[38]:


from sklearn.metrics import mean_squared_error,r2_score 
from sklearn.model_selection import cross_val_score,GridSearchCV


# In[39]:


Result =pd.DataFrame(columns=["Model","Train R2","Test RMSE ","Variance"])


# In[51]:


R2=r2_score(y_train,knn_train_pred)
R2_train=r2_score(y_train,knn_train_pred)
RMSE=np.sqrt(mean_squared_error(y_test,knn_test_pred))
Varience=R2_train-R2
Result=Result.append({"Model":"kNearest Neighbors","Train R2":R2_train,"Test R2":R2,"Test RMSE":RMSE,"Varience":Varience},ignore_index=True)
print("R2",R2)
print("RMSE:",RMSE)


# In[52]:


plt.scatter(x_train,y_train)
plt.plot(x_train,6.482+0.0541*x_train,'y')
plt.show()


# In[ ]:





# In[ ]:





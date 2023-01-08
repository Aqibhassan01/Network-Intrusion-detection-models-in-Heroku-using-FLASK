#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading and importing data 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# add the column labels
columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])


# In[3]:


df_train=pd.read_csv('KDDTrain+.txt',header=None,names=columns)
df_test=pd.read_csv('KDDTest+.txt',header=None,names=columns)


# In[4]:


df_train.head()


# In[5]:


df_train.info()


# In[6]:


print(df_train.duplicated().sum())
print(df_test.duplicated().sum())


# In[7]:


df_train.isnull().sum()


# In[8]:


df_train['attack'].value_counts()


# In[9]:


# I will convert other abnormal classes to one class

df_train["binary_attack"]=df_train.attack.map(lambda a: "normal" if a == 'normal' else "abnormal")
df_train.drop('attack',axis=1,inplace=True)

df_test["binary_attack"]=df_test.attack.map(lambda a: "normal" if a == 'normal' else "abnormal")
df_test.drop('attack',axis=1,inplace=True)


# In[10]:


df_train.select_dtypes(['object']).columns


# In[11]:


# Label Encoder
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
clm=['protocol_type', 'service', 'flag', 'binary_attack']
for x in clm:
    df_train[x]=le.fit_transform(df_train[x])
    df_test[x]=le.fit_transform(df_test[x])


# In[12]:


#Spliting the data

x_train=df_train.drop('binary_attack',axis=1)
y_train=df_train["binary_attack"]

x_test=df_test.drop('binary_attack',axis=1)
y_test=df_test["binary_attack"]


# In[13]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(x_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = x_train.columns
mutual_info.sort_values(ascending=False)


# In[14]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8));


# In[15]:


# I will choose 20 features to select
from sklearn.feature_selection import SelectKBest
sel_20_cols = SelectKBest(mutual_info_classif, k=20)
sel_20_cols.fit(x_train, y_train)
x_train.columns[sel_20_cols.get_support()]


# In[16]:


col=['service', 'flag', 'src_bytes', 'dst_bytes', 'logged_in',
       'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
       'dst_host_same_srv_rate', 'dst_host_diff_srv_rate']

x_train=x_train[col]
x_test=x_test[col]


# In[17]:


plt.figure(figsize=(12,10))
p=sns.heatmap(x_train.corr(), annot=True,cmap ='RdYlGn')  


# In[18]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)


# In[19]:


from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


# In[20]:


xgb = GradientBoostingClassifier()

param_grid = {'learning_rate':[0.7, 0.9, 1], "min_samples_split":[7,10,15,20]}


grid_clf = GridSearchCV(xgb, param_grid = param_grid, scoring = 'accuracy',cv=5)
grid_clf.fit(x_train, y_train)


# In[21]:


import pickle
# Saving model to disk
pickle.dump(grid_clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[22]:


model.predict([[20, 0.1       , 0.        , 0.        , 0.        ,
       0.04      , 0.06      , 0.03921569, 0.04      , 0.06      ]])


# In[23]:


df_train[col]


# In[ ]:





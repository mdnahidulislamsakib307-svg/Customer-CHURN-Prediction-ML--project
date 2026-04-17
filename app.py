#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression


# In[13]:


df = pd.read_csv("telecom_churn.csv")


# In[17]:


df.head(20)


# In[15]:


df.dtypes


# In[19]:


x = df.drop(['Churn','AccountWeeks','DataUsage','DayCalls','OverageFee','RoamMins'],axis=1)
y = df['Churn']


# In[22]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[23]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[24]:


numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[25]:


categorical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[26]:


preprocessor = ColumnTransformer(transformers=[
   ('num',numerical_transformer,numerical_cols),
   ('cat',categorical_transformer,categorical_cols)
])


# In[27]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[29]:


model = Pipeline(steps=[
    ('pre',preprocessor),('reg',LogisticRegression(max_iter=1000))
])


# In[30]:


model.fit(X_train,y_train)


# In[31]:


y_pred = model.predict(X_test)
print(f'{classification_report(y_pred,y_test)}')


# In[32]:


jb.dump(model,'LogisticRegression.pkl')


# In[33]:


load = jb.load('LogisticRegression.pkl')


# In[121]:


import streamlit as st
import joblib as jb
import pandas as pd  
load = jb.load('LogisticRegression.pkl')

st.title('Customer Churn Prediction')
st.write("Enter the customer details to predict if they will stay or leave.")

ContractRenewal = st.selectbox("ContractRenewal", ['0','1'])

DataPlan = st.selectbox("DataPlan", ['0','1'])
CustServCalls = st.selectbox("CustServCalls", ['0','1','2','3','4','5','6','7','8','9'])
DayMins = st.number_input("DayMins")
MonthlyCharge = st.number_input("MonthlyCharge")

if st.button("Predict Churn Status"):
    
    
    ContractRenewal = int(ContractRenewal)
    DataPlan = int(DataPlan)
    CustServCalls = int(CustServCalls)

    
    data = pd.DataFrame([{
        'ContractRenewal': ContractRenewal,
        'DataPlan': DataPlan,
        'CustServCalls': CustServCalls,
        'DayMins': DayMins,
        'MonthlyCharge': MonthlyCharge
    }])

    prediction = load.predict(data)

    if prediction[0] == 1:
        st.success("Prediction: The customer is likely to CHURN (Leave).")
    else:
        st.success("Prediction: The customer is likely to STAY (Not Churn).")


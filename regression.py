#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd                 # For data manipulation
import numpy as np                  # For numerical operations
from sklearn.model_selection import train_test_split  # To split data into train/test
from sklearn.linear_model import LinearRegression, LogisticRegression  # Regression models
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Encoding & scaling
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, confusion_matrix, root_mean_squared_error


# 

# In[116]:


df = pd.read_csv('titanic/train.csv')


# In[117]:


age_missing = df['Age'].isnull().sum()
print(age_missing)


# In[131]:


#Handling missing AGE values
age_missing = df['Age'].isnull().sum()
cabin_missing = df['Cabin'].isnull().sum()
# print(f'Missing values of: \n Age: {age_missing} \n Cabin: {cabin_missing}')
group = df.groupby('Pclass')
for Pclass, Class_df in group:
    # print(Class_df[['Pclass','Age']].head(10))
    med=Class_df['Age'].median()
    # print(med)
    mask = (df['Pclass']==Pclass) & (df['Age'].isnull())
    df.loc[mask,'Age'] = med
    # print(df.loc[df['Pclass'] == Pclass, 'Age'].head(10))
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#encoding categorical features
# df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
df.columns


# In[135]:


# linear regression predict fare
X_lin = df[['Pclass', 'SibSp', 'Parch', 'Sex_male']]
y_lin = df['Age']


# In[136]:


# logistic regression predict survived
X_log = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_male']]
y_log = df['Survived']


# In[137]:


# split dataset
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.24,random_state=42)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.24, random_state=42)
linear_model = LinearRegression()
logistic_model = LogisticRegression()

# column_names = ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_male']

# X_train_log = pd.DataFrame(X_train_log, columns=column_names)
# X_test_log = pd.DataFrame(X_test_log, columns=column_names)




# Standardization of features
scaler_lin=StandardScaler()
X_train_lin_scl = scaler_lin.fit_transform(X_train_lin)
X_test_lin_scl = scaler_lin.transform(X_test_lin)

scaler_log = StandardScaler()
X_train_log_scl = scaler_log.fit_transform(X_train_log)
X_test_log_scl = scaler_log.transform(X_test_log)





# predict 
linear_model.fit(X_train_lin, y_train_lin)
y_pred_lin = linear_model.predict(X_test_lin)

logistic_model.fit(X_train_log, y_train_log)
y_pred_log = logistic_model.predict(X_test_log)





print("LinearRegression:  ")
print("R square", r2_score(y_test_lin, y_pred_lin))
print("RMSE:", np.sqrt(mean_squared_error(y_test_lin, y_pred_lin)))
print("MAE:", mean_absolute_error(y_test_lin, y_pred_lin))





y_test_log = y_test_log[:len(X_test_log)]

print("\nLogistic Regression: ")
print("Accuracy:", accuracy_score(y_test_log, y_pred_log))
print("F1 Score:", f1_score(y_test_log, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test_log, logistic_model.predict_proba(X_test_log)[:,1]))
print("Confusion Matrix:\n", confusion_matrix(y_test_log, y_pred_log))








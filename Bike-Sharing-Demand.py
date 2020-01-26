#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# In[2]:


# import datetime
def date_diff(date):
    first_new_year=str(date[0:4])+"-01-01 00:00:00"
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    first_new_year = datetime.strptime(first_new_year, '%Y-%m-%d %H:%M:%S')
    return (date-first_new_year).days


# In[3]:


df_bike_data = pd.read_csv("bike_data.csv")


# In[4]:


df_bike_data.info()


# In[5]:


df_bike_data.describe()


# Feature engineering

# In[6]:


df_bike_data["hour"] = df_bike_data.datetime.apply(lambda x : int(x.split()[1].split(":")[0]))


# In[7]:


df_bike_data["date"] = df_bike_data.datetime.apply(lambda x : x.split()[0])


# In[8]:


df_bike_data["weekday"] = df_bike_data.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])


# In[9]:


df_bike_data['date_newyear_num']=df_bike_data["datetime"].apply(lambda x : date_diff(x))


# In[10]:


df_bike_data


# In[11]:


cols = list(df_bike_data)
cols


# In[12]:


cols.insert(len(cols), cols.pop(cols.index('count')))
cols


# In[13]:


df_bike_data = df_bike_data.loc[:, cols]


# In[14]:


df_bike_data


# Data visualization

# In[15]:


df_bike_data.groupby('date_newyear_num')['count'].mean().plot(kind='line')


# In[16]:


df_bike_data.groupby('hour')['count'].mean().plot(kind='bar')


# In[17]:


df_bike_data.groupby('season')['count'].mean().plot(kind='bar')


# In[18]:


df_bike_data.groupby('weather')['count'].mean().plot(kind='bar')


# In[19]:


df_bike_data['temp_int']=df_bike_data.temp.apply(lambda x: int(x))
df_bike_data['atemp_int']=df_bike_data.atemp.apply(lambda x: int(x))
df_bike_data['humidity_int']=df_bike_data.humidity.apply(lambda x: int(x))
df_bike_data['windspeed_int']=df_bike_data.windspeed.apply(lambda x: int(x))


# In[20]:


df_bike_data.groupby('temp_int')['count'].mean().plot(kind='line')


# In[21]:


df_bike_data.groupby('atemp_int')['count'].mean().plot(kind='line')


# In[22]:


df_bike_data.groupby('windspeed_int')['count'].mean().plot(kind='line')


# In[23]:


df_bike_data.groupby('humidity_int')['count'].mean().plot(kind='line')


# In[24]:


df_bike_data.groupby('workingday')['count'].mean().plot(kind='bar')


# In[25]:


df_bike_data.groupby('holiday')['count'].mean().plot(kind='bar')


# In[26]:


df_bike_data = df_bike_data.drop(['temp_int', 'atemp_int', 'humidity_int', 'windspeed_int'], axis=1)


# In[27]:


corrMatt = df_bike_data.corr()
corrMatt


# In[28]:


mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(15,8)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)


# The outliers

# In[29]:


fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sn.boxplot(data=df_bike_data,y="count",ax=axes[0][0])
sn.boxplot(data=df_bike_data,y="count",x="season",ax=axes[0][1])
sn.boxplot(data=df_bike_data,y="count",x="hour",ax=axes[1][0])
sn.boxplot(data=df_bike_data,y="count",x="workingday",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")


# In[34]:


df_bike_data.describe()


# In[35]:


outliers=np.abs(df_bike_data["count"]-df_bike_data["count"].mean()) >(3*df_bike_data["count"].std())


# In[36]:


outliers = pd.DataFrame(outliers)


# In[37]:


outliers


# In[38]:


df_bike_data.iloc[6658]


# In[39]:



outliers_list = outliers.loc[outliers["count"]==True]._stat_axis.values.tolist()


# In[40]:


len(outliers_list)


# In[41]:


df_bike_data = df_bike_data.drop(index=outliers_list)
df_bike_data.info()


# This is a regression problem, so if the target value follows a normal distribution, it will work well for many models.

# In[42]:


df_bike_data["count"].plot(kind = 'kde')


# This graph is not a good normal distribution. Therefore, it may be difficult to use a linear model.

# In[44]:



df_bike_data=pd.get_dummies(df_bike_data,columns=['season'])
df_bike_data=pd.get_dummies(df_bike_data,columns=['weather'])


# In[45]:


df_bike_data=pd.get_dummies(df_bike_data,columns=['weekday'])


# In[46]:


df_bike_data.columns.values


# In[48]:


df_bike_data.to_csv("bike_data_new.csv")
df_bike_data


# machine learning

# In[5]:


df = pd.read_csv("bike_data_new.csv", index_col=0)


# In[6]:


X_feature = [col for col in df.columns.values if col != 'count' and col != "datetime" and col != "date"]
y_feature = ['count']
X = df.loc[:,X_feature]
y = df.loc[:,y_feature]


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[58]:


X_train


# In[183]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from tqdm import *
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.externals import joblib


# In[184]:


def rmsle(y_test, y_pred):
    return np.sqrt(mean_squared_log_error(y_test, y_pred))

def rmse(y_test, y_pred):
    return sqrt(mean_squared_error(y_test,y_pred))


# Ridge Regression

# In[185]:


estimator = Ridge()
parameters = { 
    'alpha':[0.1, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,200]
}
rmse_scorer = metrics.make_scorer(rmse, greater_is_better=False)
grid_Ridge = GridSearchCV( estimator,
                          param_grid=parameters,
                          scoring = rmse_scorer,
                          cv=5)

grid_Ridge.fit(X=X_train,y=y_train)


# In[186]:


means = grid_Ridge.cv_results_['mean_test_score']
params = grid_Ridge.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
print(grid_Ridge.best_params_)
print(grid_Ridge.best_score_)


# In[187]:


pre=grid_Ridge.predict(X_test)
print(rmse(y_test, pre))


# In[188]:


R = Ridge(alpha = grid_Ridge.best_params_["alpha"])
R.fit(X=X_train,y=y_train)


# In[261]:


pre=R.predict(X_test)
print("RMSE of Ridge Regression: ", rmse(y_test, pre))


# In[262]:


x_axix = list(range(100))
plt.figure(figsize=(10,8))
plt.title('Ridge Regression')
plt.plot(x_axix, pre[0:100], color='green', label='Prediction')
plt.plot(x_axix, y_test[0:100], color='red', label='Reality')
plt.xlabel('Sample index')
plt.ylabel('Count')
plt.show()


# In[190]:


with open('Ridge.pkl', 'wb') as f:
    joblib.dump(R, 'Ridge.pkl')


# Lasso Regression

# In[191]:


estimator = Lasso()
parameters = { 
    'alpha':[0.1, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,200]
}
rmse_scorer = metrics.make_scorer(rmse, greater_is_better=False)
grid_Lasso = GridSearchCV( estimator,
                          param_grid=parameters,
                          scoring = rmse_scorer,
                          cv=5)

grid_Lasso.fit(X=X_train,y=y_train)


# In[192]:


means = grid_Lasso.cv_results_['mean_test_score']
params = grid_Lasso.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
print(grid_Lasso.best_params_)
print(grid_Lasso.best_score_)


# In[193]:


pre=grid_Lasso.predict(X_test)


# In[194]:


rmse(y_test, pre)


# In[195]:


L = Lasso(alpha = grid_Lasso.best_params_["alpha"])
L.fit(X=X_train,y=y_train)


# In[263]:


pre=L.predict(X_test)
print("RMSE of Lasso Regression: ", rmse(y_test, pre))


# In[264]:


x_axix = list(range(100))
plt.figure(figsize=(10,8))
plt.title('Lasso Regression')
plt.plot(x_axix, pre[0:100], color='green', label='Prediction')
plt.plot(x_axix, y_test[0:100], color='red', label='Reality')
plt.xlabel('Sample index')
plt.ylabel('Count')
plt.show()


# In[197]:


with open('Lasso.pkl', 'wb') as f:
    joblib.dump(L, 'Lasso.pkl')


# GradientBoostingRegressor

# In[198]:


estimator = GradientBoostingRegressor()
parameters = { 
    'n_estimators':[100,500,1000],
    'learning_rate': [0.1,0.05,0.02],
    'max_depth':[4,3,2],
    'min_samples_leaf':[1,2,3]
}
rmse_scorer = metrics.make_scorer(rmse, greater_is_better=False)
grid_GBR = GridSearchCV( estimator,
                          param_grid=parameters,
                          scoring = rmse_scorer,
                          cv=5)

grid_GBR.fit(X=X_train,y=y_train.to_numpy().ravel())


# In[199]:


means = grid_GBR.cv_results_['mean_test_score']
params = grid_GBR.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
print(grid_GBR.best_params_)
print(grid_GBR.best_score_)


# In[200]:


pre=grid_GBR.predict(X_test)


# In[201]:


rmse(y_test, pre)


# In[202]:


GBR = GradientBoostingRegressor(n_estimators = grid_GBR.best_params_["n_estimators"],
                               learning_rate = grid_GBR.best_params_["learning_rate"],
                               max_depth = grid_GBR.best_params_["max_depth"],
                               min_samples_leaf = grid_GBR.best_params_["min_samples_leaf"])
GBR.fit(X=X_train,y=y_train)


# In[265]:


pre=GBR.predict(X_test)
print("RMSE of Gradient Boosting Regression: ", rmse(y_test, pre))


# In[266]:


x_axix = list(range(100))
plt.figure(figsize=(10,8))
plt.title('Gradient Boosting Regression')
plt.plot(x_axix, pre[0:100], color='green', label='Prediction')
plt.plot(x_axix, y_test[0:100], color='red', label='Reality')
plt.xlabel('Sample index')
plt.ylabel('Count')
plt.show()


# In[204]:


with open('GBR.pkl', 'wb') as f:
    joblib.dump(GBR, 'GBR.pkl')


# Random Forest Regressor

# In[271]:


estimator = RandomForestRegressor()
parameters = { 
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 1000, 2000],
    'random_state' : [0],
    'n_jobs' : [-1]
}
rmse_scorer = metrics.make_scorer(rmse, greater_is_better=False)
grid_RFR = GridSearchCV( estimator,
                          param_grid=parameters,
                          scoring = rmse_scorer,
                          cv=5)

grid_RFR.fit(X=X_train,y=y_train.to_numpy().ravel())


# In[272]:


means = grid_RFR.cv_results_['mean_test_score']
params = grid_RFR.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
print(grid_RFR.best_params_)
print(grid_RFR.best_score_)


# In[273]:


pre=grid_RFR.predict(X_test)
rmse(y_test, pre)


# In[274]:


RFR = RandomForestRegressor(min_samples_leaf = grid_RFR.best_params_["min_samples_leaf"],
                            min_samples_split = grid_RFR.best_params_["min_samples_split"],
                            n_estimators = grid_RFR.best_params_["n_estimators"],
                            random_state=0, 
                            n_jobs=-1)
RFR.fit(X=X_train,y=y_train.to_numpy().ravel())


# In[275]:


pre=RFR.predict(X_test)
print("RMSE of Random Forest Regression: ", rmse(y_test, pre))


# In[276]:


x_axix = list(range(100))
plt.figure(figsize=(10,8))
plt.title('Random Forest Regression')
plt.plot(x_axix, pre[0:100], color='green', label='Prediction')
plt.plot(x_axix, y_test[0:100], color='red', label='Reality')
plt.xlabel('Sample index')
plt.ylabel('Count')
plt.show()


# In[277]:


with open('RFR.pkl', 'wb') as f:
    joblib.dump(RFR, 'RFR.pkl')


# In[2]:


GBR = joblib.load("GBR.pkl")


# The most important features

# In[3]:


GBR.feature_importances_


# In[7]:


X_feature


# In[10]:


indies = np.argsort(GBR.feature_importances_, kind='heapsort')[::-1]


# In[11]:


indies


# In[12]:


for index in indies:
    print({X_feature[index]:GBR.feature_importances_[index]})


# In[42]:


list_importances = [{X_feature[index]:GBR.feature_importances_[index]} for index in indies]
keys = [list(list_importances[i].keys())[0] for i in range(len(list_importances))]
values = np.sort(GBR.feature_importances_)[::-1]


# In[45]:



plt.figure(figsize=(15,4))
plt.title('10 Most Important Features')
plt.bar(keys[0:10], values[0:10])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()


# In[ ]:





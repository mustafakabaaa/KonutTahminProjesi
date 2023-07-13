# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:07:24 2022

@author: musto
"""

import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn import model_selection

df_2 = pd.read_excel(r"son_hali_z.xlsx")

#df_2.drop("Unnamed: 0", axis = 1, inplace = True)
df = df_2.copy()

df.columns


df=df[[ 'yas', 'Bina_kat_sayisi', 'net_alan', 'oda_sayisi',
       'kat', 'isinma', 'ilce', 'fiyat']]

X = df.drop(["fiyat"], axis = 1)
y = df["fiyat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 144)

params = {"colsample_bytree":[0.4,0.5,0.6],
         "learning_rate":[0.01,0.02,0.09],
         "max_depth":[2,3,4,5,6],
         "n_estimators":[100,200,500,2000]}

xgb = XGBRegressor()

grid = GridSearchCV(xgb, params, cv = 10, n_jobs = -1, verbose = 2)
grid.fit(X_train, y_train)  

grid.best_params_

xgb1 = XGBRegressor(colsample_bytree = 0.5, learning_rate = 0.09, max_depth = 4, n_estimators = 2000)


model_xgb = xgb1.fit(X_train, y_train)

model_xgb.predict(X_test)[15:20]

model_xgb.score(X_test, y_test)


model_xgb.score(X_train, y_train)

np.sqrt(-1*(cross_val_score(model_xgb, X_test, y_test, cv=10, scoring='neg_mean_squared_error'))).mean()

importance = pd.DataFrame({"Importance": model_xgb.feature_importances_},
                         index=X_train.columns)

importance

from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate an xgboost regression model on the housing dataset
from numpy import absolute
scores = cross_val_score(xgb, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

import matplotlib.pyplot as plt

plt.scatter(df["yas"],df["fiyat"])
plt.ylabel("fiyat")
plt.show();

preds=xgb1.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

confusion_matrix(y_test, preds)
accuracy_score(y_test, preds)

mse=mean_squared_error(y_test, preds)
print("mse:",mse)
print("Rmse:",mse*(1/2.0))

x_ax=range(len(y_test))
plt.plot(x_ax, y_test,label="orjinal")
plt.plot(x_ax, preds,label="tahmin")
plt.legend()
plt.show()

ypred=xgb1.predict(X_test)
cm=confusion_matrix(y_test, preds)
print(cm)
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

from sklearn.metrics import mean_squared_log_error
msle=mean_squared_error(y_test, preds)
print(    msle)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test, preds)
print('MAE:%f'% mae)
from sklearn.metrics import r2_score
r2=r2_score(y_test, preds)
print(r2)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, preds)
print(mse)

me=max_error(y_test, preds)
print(me)
cr=explained_variance_score(y_test, preds)
print(cr)
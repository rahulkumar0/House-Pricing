import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

#removing outliers/abnormal data
plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.show()
train=train.drop(train[train['GrLivArea']>4000].index)

#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train[train['Electrical'].isnull()].index)

test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)
total1 = test.isnull().sum().sort_values(ascending=False)
missing_data1 = pd.DataFrame({'Index':total1.index,'total':total1.values})
dic = []
for i in missing_data1['Index']:
    dic.append(i)

for i in dic:
    test = test.drop(test[test[i].isnull()].index)

#make numeric(actually categorial) categorial
train['MSSubClass'] = train['MSSubClass'].apply(str)
train['OverallCond'] = train['OverallCond'].astype(str)
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)

#encode categoial variable
train = pd.get_dummies(train)

#X_train y_train
X_train = train.loc[:,train.columns!='Id']
X_train = X_train.loc[:,X_train.columns!='SalePrice'].values
y_train = train['SalePrice'].values

# Fitting Multiple Linear Regression to the Training set
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
regressor1 = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
regressor2 =LinearRegression()
regressor3 = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor4 = SVR(kernel = 'rbf')

# Defining k for cross validation
from sklearn.model_selection import cross_val_score
def validation(model):
    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
    return accuracies

#applying validation function
accuracy = validation(regressor1)
print(accuracy.mean(),accuracy.std())

accuracy = validation(regressor2)
print(accuracy.mean(),accuracy.std())
accuracy = validation(regressor3)
print(accuracy.mean(),accuracy.std())
accuracy = validation(regressor4)
print(accuracy.mean(),accuracy.std())
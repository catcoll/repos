#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:09:23 2024

@author: catcoll
    
in this code:
    uses lazy regressor to fit many regressors at the same time to assess parameter/feature tuning,
    hyper parameter tuning with best regressor identified from lazy regressor and other high performers,
    compare performance of base model and grid searched best estimator with AIC metric
    
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
import time
from lazypredict.Supervised import LazyRegressor
from sklearn.neighbors import KNeighborsRegressor

#%% load excel files with dataframe from residual (created in random forest code)


file = '/Users/catcoll/Documents/spatial_data/OLS_converted_scaled.xlsx'
data = pd.read_excel(file)

#%% set up train-test split
#feature columns
# X = data[['numerical_feature_0', 'numerical_feature_1', 'one_hot_11',
#        'one_hot_21', 'one_hot_31', 'one_hot_41', 'one_hot_51', 'one_hot_71',
#        'one_hot_81', 'one_hot_90']]

# labels = ["canopy", "imper", 'dist to coast', 'open_water', 'urban', 
#           'barren', 'forest', 'scrub', 'moss/grass',
#           'agriculture','wetlands']


X = data[['numerical_feature_0', 'numerical_feature_1', 'numerical_feature_2', 'one_hot_51']]


#target column
y = data['residual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1066)


reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)
# best model is KNieghborsRegressor with all numerical data and scrub land type data

#%% hyperparameter tuning

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


# Hyperparameter Tuning with Grid Search
#KNeighbors regressor
param_grid = {
    'n_neighbors': [28, 29, 30, 31, 32, 33, 34, 35, 36],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [20, 30, 40],
    'p': [1,2]
}

grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best RMSE from Grid Search: {np.sqrt(-grid_search.best_score_)}')

#train models: base model and best estimator from grid search
model = KNeighborsRegressor()
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model2 = grid_search.best_estimator_
model2 = model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

# hist gb

# Hyperparameter Tuning with Grid Search
# hist Gradient boosting
param_grid2 = {
    'loss': ['squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile'],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1, 2],
    'max_iter': [25, 50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [0.5, 0.7, 1],
    'max_bins': [255, 512, 1024],
    'l2_regularization': [0.0, 0.1, 0.5, 1.0],
}


grid_search2 = GridSearchCV(HistGradientBoostingRegressor() , param_grid2, cv=5, scoring='neg_root_mean_squared_error', verbose=1)
grid_search2.fit(X_train, y_train)
print(f'Best parameters: {grid_search2.best_params_}')
print(f'Best RMSE from Grid Search: {np.sqrt(-grid_search2.best_score_)}')


#train models: base model and best estimator from grid search
model3 = HistGradientBoostingRegressor()
model3 = model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)

model4 = grid_search2.best_estimator_
model4 = model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)

#Code to calculate AIC
#add constant
p=1+1

def llf_(y, X, pr):
    # return maximized log likelihood
    nobs = float(X.shape[0])
    nobs2 = nobs / 2.0
    nobs = float(nobs)
    resid = y - pr
    ssr = np.sum((resid)**2)
    llf = -nobs2*np.log(2*np.pi) - nobs2*np.log(ssr / nobs) - nobs2
    return llf

def aic(y, X, pr, p):
    # return aic metric
    llf = llf_(y, X, pr)
    return -2*llf+2*p

#Evaluate model perfromance
print(aic(y_test, X_test, y_pred, p))
print(aic(y_test, X_test, y_pred2, p))
print(aic(y_test, X_test, y_pred3, p))
print(aic(y_test, X_test, y_pred4, p))


# Calculate RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
print(f'Test RMSE with Best Model: {test_rmse}')
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'Test R-Squared: {r2}')

#%% Testing more models

# Hyperparameter Tuning with Grid Search
# gradient boositng
param_grid3 = {
    "loss":['absolute_error', 'squared_error', 'huber', 'quantile'],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    # "min_samples_split": np.linspace(0.1, 0.5, 12),
    # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    # "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }

grid_search3 = GridSearchCV(GradientBoostingRegressor() , param_grid3, cv=5, scoring='neg_root_mean_squared_error', verbose=1)
grid_search3.fit(X_train, y_train)
print(f'Best parameters: {grid_search3.best_params_}')
print(f'Best RMSE from Grid Search: {np.sqrt(-grid_search3.best_score_)}')



model5 = GradientBoostingRegressor()
model5 = model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_test)

model6 = grid_search3.best_estimator_
model6 = model6.fit(X_train, y_train)
y_pred6 = model6.predict(X_test)


#feature importance of model
import matplotlib.pyplot as plt

labels = ["canopy", "imper",'dist to coast']


# Feature importances from the best model
feature_importance = model5.feature_importances_

# Create a DataFrame with custom labels
importance_df = pd.DataFrame({
    'features': labels,  # Use custom labels here
    'importance': feature_importance
})

# Sort values by importance
importance_df.sort_values(by='importance', ascending=False, inplace=True)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['features'], importance_df['importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Best Gradient Boosting Regressor Winter')
plt.annotate("Best parameters: \nlearning_rate: 0.01, \n max_depth: 3, \n n_estimators: 200", xy=(0.3,6))
plt.gca().invert_yaxis()
plt.show()









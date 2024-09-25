#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:11:33 2024

@author: catcoll

Gradient Boosting:
    
    trains gradient boosting regression and plots feature importance
    code to perform early stopping
    hyper parameter tuning with grid search and various model assessment metrics
"""



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import time

#%% load excel files with dataframe from residual (created in random forest code)


file = '/Users/catcoll/Documents/spatial_data/high_pass_h2o_removed.xlsx'
data = pd.read_excel(file)

#%% set up train-test split

#change distance to coast feature to categorical (all values between 0-1 are True)
#comment out when using continuous distance to coast feature
# data.columns
coast = data[['numerical_feature_2']]
c=coast.values
binary_coast=(c<1).astype(int)
new_data=data.drop(['numerical_feature_2'], axis = 1)
new_data.columns
new_data.insert(2, 'numerical_feature_2', binary_coast)
data=new_data


X = data[['numerical_feature_0', 'numerical_feature_1', 'numerical_feature_2', 'one_hot_11',
       'one_hot_21', 'one_hot_31', 'one_hot_41', 'one_hot_51', 'one_hot_71',
       'one_hot_81', 'one_hot_90']]

#target column
y = data['residual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1066)


#%% fit model (sklearn)


model = GradientBoostingRegressor(learning_rate=0.01, max_depth= 5, n_estimators=100)
model.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
rmse
from sklearn.metrics import mean_squared_error, r2_score
# Making predictions on the same data or new data
predictions = model.predict(X_test)
 
# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

#%% earyl stopping
# params = dict(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42)

# gbm_full = GradientBoostingRegressor(**params)
# gbm_early_stopping = GradientBoostingRegressor(
#     **params,
#     validation_fraction=0.1,
#     n_iter_no_change=10,
# )

# start_time = time.time()
# gbm_full.fit(X_train, y_train)
# training_time_full = time.time() - start_time
# n_estimators_full = gbm_full.n_estimators_

# start_time = time.time()
# gbm_early_stopping.fit(X_train, y_train)
# training_time_early_stopping = time.time() - start_time
# estimators_early_stopping = gbm_early_stopping.n_estimators_

# train_errors_without = []
# val_errors_without = []

# train_errors_with = []
# val_errors_with = []

# for i, (train_pred, val_pred) in enumerate(
#     zip(
#         gbm_full.staged_predict(X_train),
#         gbm_full.staged_predict(X_test),
#     )
# ):
#     train_errors_without.append(mean_squared_error(y_train, train_pred))
#     val_errors_without.append(mean_squared_error(y_test, val_pred))

# for i, (train_pred, val_pred) in enumerate(
#     zip(
#         gbm_early_stopping.staged_predict(X_train),
#         gbm_early_stopping.staged_predict(X_test),
#     )
# ):
#     train_errors_with.append(mean_squared_error(y_train, train_pred))
#     val_errors_with.append(mean_squared_error(y_test, val_pred))

# fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

# axes[0].plot(train_errors_without, label="gbm_full")
# axes[0].plot(train_errors_with, label="gbm_early_stopping")
# axes[0].set_xlabel("Boosting Iterations")
# axes[0].set_ylabel("MSE (Training)")
# axes[0].set_yscale("log")
# axes[0].legend()
# axes[0].set_title("Training Error")

# axes[1].plot(val_errors_without, label="gbm_full")
# axes[1].plot(val_errors_with, label="gbm_early_stopping")
# axes[1].set_xlabel("Boosting Iterations")
# axes[1].set_ylabel("MSE (Validation)")
# axes[1].set_yscale("log")
# axes[1].legend()
# axes[1].set_title("Validation Error")

# training_times = [training_time_full, training_time_early_stopping]
# labels = ["gbm_full", "gbm_early_stopping"]
# bars = axes[2].bar(labels, training_times)
# axes[2].set_ylabel("Training Time (s)")

# for bar, n_estimators in zip(bars, [n_estimators_full, estimators_early_stopping]):
#     height = bar.get_height()
#     axes[2].text(
#         bar.get_x() + bar.get_width() / 2,
#         height + 0.001,
#         f"Estimators: {n_estimators}",
#         ha="center",
#         va="bottom",
#     )

# plt.tight_layout()
# plt.show()


#%% feature importance

labels = ["canopy", "imper", 'dist to coast','open_water', 'urban', 
          'barren', 'forest', 'scrub', 'moss/grass',
          'agriculture','wetlands']

feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'features': X_train.columns,
                              'importance': feature_importance})
importance_df.sort_values(by='importance', ascending=False, inplace=True)
importance_df

#
importances = model.feature_importances_

labels = ["canopy", "imper", 'dist to coast','open_water', 'urban', 
          'barren', 'forest', 'scrub', 'moss/grass',
          'agriculture','wetlands']

feature_importances_data = pd.DataFrame({
    'Feature': labels,
    'Importance': importances
    })

feature_importances_data = feature_importances_data.sort_values(by='Importance', ascending=False)

print(feature_importances_data)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_data['Feature'], feature_importances_data['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Gradient Boosting (high pass h2o removed, binary coast)')
plt.gca().invert_yaxis()
plt.show()


#%% XGBoost

import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

dtrain_reg = xgb.DMatrix(X_train, y_train)
dtest_reg = xgb.DMatrix(X_test, y_test)

# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}
n=100

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
)

preds = model.predict(dtest_reg)
rmse = root_mean_squared_error(y_test, preds)


#%% tuning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import time

# Hyperparameter Tuning with Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10, 20 ,30 ,40 ,50],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5, scoring='explained_variance', verbose=1)
grid_search.fit(X_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best RMSE from Grid Search: {np.sqrt(-grid_search.best_score_)}')

# Feature Importance Plot
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'features': X_train.columns, 'importance': feature_importance})
importance_df.sort_values(by='importance', ascending=False, inplace=True)
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['features'], importance_df['importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Gradient Boosting Regressor')
plt.gca().invert_yaxis()
plt.show()

# Modelfit Function for Regression
def modelfit(alg, X_train, y_train, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(X_train[predictors], y_train)
    
    # Predict training set
    dtrain_predictions = alg.predict(X_train[predictors])
    
    # Perform cross-validation
    if performCV:
        cv_score = cross_val_score(alg, X_train[predictors], y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    
    # Print model report
    print("\nModel Report")
    print(f"Training RMSE : {np.sqrt(mean_squared_error(y_train, dtrain_predictions)):.4g}")
    
    if performCV:
        print(f"CV RMSE : Mean - {np.sqrt(-np.mean(cv_score)):.4g} | Std - {np.sqrt(np.std(cv_score)):.4g} | Min - {np.sqrt(-np.min(cv_score)):.4g} | Max - {np.sqrt(-np.max(cv_score)):.4g}")
        
    # Print Feature Importance
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()

# Choose all predictors
predictors = X_train.columns.tolist()
gbm0 = GradientBoostingRegressor(random_state=10)
modelfit(gbm0, X_train, y_train, predictors)



from sklearn.metrics import mean_squared_error

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE with Best Model: {test_rmse}')


import matplotlib.pyplot as plt

# Feature importances from the best model
feature_importance = best_model.feature_importances_

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


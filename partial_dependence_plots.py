#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:14:24 2024

@author: catcoll

Partial Dependence Plots

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

#%% fit model

file = '/Users/catcoll/Documents/spatial_data/OLS_winter.xlsx'
data = pd.read_excel(file)
data.columns

#feature columns
# X = data[['numerical_feature_0', 'numerical_feature_1', 'numerical_feature_2','one_hot_11',
#        'one_hot_21', 'one_hot_31', 'one_hot_41', 'one_hot_51', 'one_hot_71',
#        'one_hot_81', 'one_hot_90']]
X = data[['numerical_feature_0', 'numerical_feature_1', 'numerical_feature_2','one_hot_51']]

#target column
y = data['residual']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#%%

numerical_features = ['numerical_feature_0', 'numerical_feature_1', 'numerical_feature_2']
categorical_features = X_train.columns.drop(numerical_features)



features_info = {
    # features of interest
    "features": ['numerical_feature_2', 'numerical_feature_0', 'numerical_feature_1','one_hot_51'],
    # type of partial dependence plot
    "kind": "average",
    # information regarding categorical features
    "categorical_features": categorical_features,
}
_, ax = plt.subplots(ncols=2, nrows=2, figsize=(9, 8), constrained_layout=True)
display = PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    **features_info,
    ax=ax,
    
)
display.figure_.suptitle(
    (
        "Partial dependence for RF Regressor"
    ),
    fontsize=16,
)











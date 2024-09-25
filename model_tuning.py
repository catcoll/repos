#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:11:39 2024

@author: catcoll

A Data Driven Approach to Model Selection

source: https://machinelearningmastery.com/create-algorithm-test-harness-scratch-python/

Supervised learning (must have target var for testing)

"""






#%% Step 1: Test Harness


"""
A test harness provides a consistent way to evaluate machine learning algorithms on a dataset.

It involves 3 elements:

The resampling method to split-up the dataset.
The machine learning algorithm to evaluate.
The performance measure by which to evaluate predictions.

"""




import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV




#%% hyper parameter tuning (optional) and model evaluation


def evaluate_algorithms(X_train, y_train, algorithms, param_grids=None, cv_folds=5, scoring='neg_mean_squared_error', perform_grid_search=True):
    results = []

    if param_grids is None and perform_grid_search:
        raise ValueError("Parameter grids must be provided if performing grid search.")
    
    for i, model in enumerate(algorithms):
        print(f"Evaluating {model.__class__.__name__}...")

        if perform_grid_search:
            # Check if a parameter grid is provided for the model
            if param_grids and i < len(param_grids):
                param_grid = param_grids[i]
            else:
                print(f"No parameter grid provided for {model.__class__.__name__}. Skipping grid search.")
                continue
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_folds, scoring=scoring, verbose=1, n_jobs=-1 )
            
            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)
            
            # Get the best parameters and score
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert from negative MSE to positive MSE
            best_rmse = np.sqrt(best_score)
            
            # Predict using the best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_train)
            
            # Calculate R-squared
            best_r2 = r2_score(y_train, y_pred)
            
            # Store the results
            results.append({
                'model': model.__class__.__name__,
                'best_params': best_params,
                'best_rmse': best_rmse,
                'best_r2': best_r2
            })
            
            print(f"Best parameters for {model.__class__.__name__}: {best_params}")
            print(f"Best RMSE for {model.__class__.__name__}: {best_rmse}")
            print(f"Best R^2 for {model.__class__.__name__}: {best_r2}\n")
        else:
            # Train model with default parameters
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            r2 = r2_score(y_train, y_pred)
            
            # Store the results
            results.append({
                'model': model.__class__.__name__,
                'best_params': 'Default',
                'best_rmse': rmse,
                'best_r2': r2
            })
            
            print(f"Default RMSE for {model.__class__.__name__}: {rmse}")
            print(f"Default R^2 for {model.__class__.__name__}: {r2}\n")
    
    return results



#%% test algs

# load excel files with dataframe from residual (created in random forest code)


file = '/Users/catcoll/Documents/spatial_data/OLS_summer.xlsx'
data = pd.read_excel(file)

# set up train-test split
#feature columns
# X = data[['numerical_feature_0', 'numerical_feature_1', 'one_hot_11',
#        'one_hot_21', 'one_hot_31', 'one_hot_41', 'one_hot_51', 'one_hot_71',
#        'one_hot_81', 'one_hot_90']]

#parameters that seem to return the best metrics
X = data[['numerical_feature_0', 'numerical_feature_1', 'one_hot_51']]
#target column
y = data['residual']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1066)



# Define the algorithms to test
models = [
    KNeighborsRegressor
]

# Define the parameter grids, match with regressor in models list by ordering

param_grids= {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [20, 30, 40],
    'p': [1,2]
}

# Evaluate with grid search
results_with_grid_search, best_model1 = evaluate_algorithms(
    X_train, y_train, models, param_grids=param_grids, perform_grid_search=True
)

# Evaluate without grid search (just test default models)
results_without_grid_search, best_model2 = evaluate_algorithms(
    X_train, y_train, models, perform_grid_search=False
)

# Print results for models with grid search
print("Results with Grid Search:")
for result in results_with_grid_search:
    print(f"Model: {result['model']}")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Best RMSE: {result['best_rmse']}")
    print(f"Best r2: {result['best_r2']}")
    print("-" * 30)

# Print results for models without grid search
print("Results without Grid Search:")
for result in results_without_grid_search:
    print(f"Model: {result['model']}")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Best RMSE: {result['best_rmse']}")
    print(f"Best r2: {result['best_r2']}")
    print("-" * 30)

#%% plot feature importance



labels = ["canopy", "imper",'open_water', 'urban', 
          'barren', 'forest', 'scrub', 'moss/grass',
          'agriculture','wetlands']

import matplotlib.pyplot as plt

# Feature importances from the best model
feature_importance = models[5].feature_importances_

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
plt.title(str(models[5]))
plt.gca().invert_yaxis()
plt.show()




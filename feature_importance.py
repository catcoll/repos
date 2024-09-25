#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:31:03 2024

@author: catcoll
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt




file = '/Users/catcoll/Documents/urban_summary.xlsx'
data = pd.read_excel(file)
data.columns

X = data[['open_water', 'urban', 'wetlands', 'forest', 'agriculture','scrub', 'high', 'med', 'low', 'imper', 'canopy',]]
y = data['mean_res']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

importances = model.feature_importances_

feature_importances_data = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
    })

feature_importances_data = feature_importances_data.sort_values(by='Importance', ascending=False)

print(feature_importances_data)



# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_data['Feature'], feature_importances_data['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Random Forest Regressor')
plt.gca().invert_yaxis()
plt.show()



# predictions = model.predict(X_test)


# mse = mean_squared_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)
# mse

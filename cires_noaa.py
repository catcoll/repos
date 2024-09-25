#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:01:57 2024

@author: catcoll
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


# read in data with appropriate delimiter
ozone=pd.read_csv(r'/Users/catcoll/Downloads/dobson_toMLO.txt',
                  sep='\s+', skiprows=1)

#create datetime column for time series processing
ozone['datetime'] = pd.to_datetime(ozone['Local_Date']+' '+ozone['Time'])
ozone['datetime']

# Drop rows where elements are NaN
ozone = ozone.dropna(how='all')

#summarize
ozone['Total_Ozone'].describe()


#%% Annual means

# Extract year from Local_Date
ozone['Year'] = ozone['datetime'].dt.year

# Group by year and calculate the mean for each year
annual_avg = ozone.groupby('Year')['Total_Ozone'].mean().reset_index()

# Rename columns for clarity
annual_avg.columns = ['Year', 'Average_Total_Ozone']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(annual_avg['Year'], annual_avg['Average_Total_Ozone'], marker='o', linestyle='-')
plt.title('Annual Average of Total Ozone')
plt.xlabel('Year')
plt.ylabel('Average Total Ozone')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

annual_avg['Average_Total_Ozone'].describe()
#%% calculate linear trend
# Prepare data for linear regression
X = annual_avg['Year'].values.reshape(-1, 1)
y = annual_avg['Average_Total_Ozone'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict using the model
predictions = model.predict(X)

# Plotting
plt.figure(figsize=(12, 6))

# Plot the annual average data
plt.plot(annual_avg['Year'], annual_avg['Average_Total_Ozone'], marker='o', linestyle='-', label='Annual Average')

# Plot the trend line
plt.plot(annual_avg['Year'], predictions, color='red', linestyle='--', label='Trend Line')

plt.title('Annual Average of Total Ozone with Trend Line')
plt.xlabel('Year')
plt.ylabel('Average Total Ozone')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#%% Polynomial regression
X = annual_avg[['Year']]
y = annual_avg['Average_Total_Ozone']

ozone['Year'] = ozone['datetime'].dt.year
ozone['Year'] = ozone['Year'].astype(int)  # Ensure Year is of integer type
# Define the degree of the polynomial features
degree = 3

# Create polynomial features
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Create a range of values for plotting the polynomial regression curve
X_plot = np.linspace(X['Year'].min(), X['Year'].max(), 500).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

# Plotting
plt.figure(figsize=(12, 6))

# Plot actual data points
plt.scatter(X, y, color='blue', label='Actual Data')

# Plot polynomial regression curve
plt.plot(X_plot, y_plot, color='red', label=f'Polynomial Regression (Degree {degree})')

plt.title('Polynomial Regression of Total Ozone Over Years')
plt.xlabel('Year')
plt.ylabel('Total Ozone')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% seasonal means


df = pd.DataFrame(ozone)

# Convert 'datetime' column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Set 'datetime' as the index
df.set_index('datetime', inplace=True)

# Function to determine season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

# Add 'season' and 'year' columns
df['season'] = pd.DatetimeIndex(df.index).month.map(get_season)
df['year'] = df.index.year

# Compute seasonal means for each year and season
seasonal_means_by_year = df.groupby(['year', 'season'])['Total_Ozone'].mean().unstack()
#Summary stats
seasonal_means_by_year[['Fall', 'Summer','Spring','Winter']].describe()

# Plot seasonal means
plt.figure(figsize=(12, 8))

# Define seasons and colors for plotting
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
colors = ['c', 'g', 'orange', 'brown']


for i, season in enumerate(seasons):
    # Extract seasonal means for the season
    if season in seasonal_means_by_year.columns:
        seasonal_means = seasonal_means_by_year[season]
        years = seasonal_means.index
        means = seasonal_means.values
        
        # Plot seasonal means
        plt.plot(years, means, marker='o', linestyle='-', color=colors[i], label=season)
        


plt.xlabel('Year')
plt.ylabel('Average Total Ozone')
plt.title('Seasonal Averages of Total Ozone by Year with Trend Lines')
plt.legend(title='Season')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Decompose time series


#format data, must be pandas dataframe
decomp=pd.DataFrame(ozone)
#make sure datetime is in datetime format
decomp['datetime'] = pd.to_datetime(decomp['datetime'])
# set datetime as the index
decomp.set_index('datetime', inplace=True)

#decompose, additive or mulplicative model
decomposition = seasonal_decompose(decomp['Total_Ozone'], model='mulplicative', period=365)
decomposition.plot()

#annual average
#format data, must be pandas dataframe
decomp2=pd.DataFrame(annual_avg)
#make sure datetime is in datetime format
decomp2['Year'] = pd.to_datetime(decomp2['Year'])
# set datetime as the index
decomp2.set_index('Year', inplace=True)

#decompose, additive or mulplicative model
decomposition = seasonal_decompose(decomp2['Average_Total_Ozone'], model='mutiplicative', period=1)
decomposition.plot()


from matplotlib import pyplot
from math import sqrt

#try other models, ARIMA

decomp.index = decomp.index.to_period('Y')

model = ARIMA(decomp['Total_Ozone'], order=(5,1,0))

model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()

# density plot of residuals
residuals.plot(kind='kde')

# summary stats of residuals
print(residuals.describe())

# split into train and test sets
X = decomp['Total_Ozone'].values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
 model = ARIMA(history, order=(5,1,0))
 model_fit = model.fit()
 output = model_fit.forecast()
 yhat = output[0]
 predictions.append(yhat)
 obs = test[t]
 history.append(obs)
 print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.title("ARIMA Total Ozone RMSE = 7.763")
pyplot.show()



test.shape






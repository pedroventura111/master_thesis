import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as LinRegr
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D

# File Location
datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_10_PIPS_Ticks_Bid_2022.01.01_2022.12.31.csv' # Renko 10 - 1 year set

# Type of Detector - must be given by the user
slope_detector = False
error_detector = False
slope_and_error_detector = True

# Error and Slopes Threshold
max_error = 1e-5
min_slope = -5e-4
max_slope = 5e-4

# Plot the results
def PlotLinearRegression(xx, yy_pred, color):
    
    plt.plot(xx, yy_pred, color,  linewidth = 2.5) 
	
    legend_lines = [Line2D([0], [0], color = 'blue', lw = 4),
			Line2D([0], [0], color = 'green', lw = 4),
			Line2D([0], [0], color = 'grey', lw = 4),               
			Line2D([0], [0], color = 'red', lw = 4),
	]
	
    plt.legend(legend_lines, ['Financial Data', 'Linear Section', 'Accepted Slope but Declined Error','Non-Linear Section'])
    plt.xlabel('Samples', color='black')
    plt.ylabel('Forex Data', color='black')

# Linear Regression: xx = indexes and yy = values
def LinearRegression(xx, yy):

    params = {'n_jobs': -1}

    linear_regression = LinRegr(**params).fit(xx, yy)
    yy_pred = linear_regression.predict(xx)

    # y = mx + b
    m = linear_regression.coef_
    #b = linear_regression.intercept_

    mse = mean_squared_error(yy, yy_pred)
    # print("Linear Regression MSE = " + mse.astype(str))

    return mse, m, yy_pred

# Performs the Linearity Detector
def LinearityDetector(begin, end):

    color = 'red'

    # Computes the linear regression and check if the slope is in the defined boundaries
    if slope_detector:
        # Compute Linear Regression
        ___, slope, yy_pred = LinearRegression(indexes[begin:end], input_data[begin:end])

        if (slope > min_slope) and (slope < max_slope):
            print("Accepted Slope")
            color = 'green'
        else:
            print("Declined Slope")

    # Computes the linear regression and check if the error is lower than the threshold
    elif error_detector:
        # Compute Linear Regression
        mse, ___, yy_pred = LinearRegression(indexes[begin:end], input_data[begin:end])
        print(mse)

        if (mse < max_error):
            print("Accepted Error")
            color = 'green'
        else:
            print("Declined Error")

    # Computes the linear regression, checks if the slope is in between the defined boundaries, and then checks if the error is enoughly low to be accepted
    elif slope_and_error_detector:
        # Compute Linear Regression
        mse, slope, yy_pred = LinearRegression(indexes[begin:end], input_data[begin:end])

        if (slope > min_slope) and (slope < max_slope):
            if (mse < max_error):
                print("Accepted Slope and Error")
                color = 'green'
            else:
                print("Accepted Threshold but Denied Error")
                color = 'grey'
        else:
            print("Declined Slope")                

    PlotLinearRegression(indexes[begin:end], yy_pred, color)

# Dataframe with data from the csv file
df = pd.read_csv(datalocation, names = ['Time (EET)', 'EndTime', 'Open', 'High', 'Low', 'Close', 'Volume'], skiprows = 1)

# Verifies missing values in the csv file and deletes the corresponding row of the csv file
if (df.Open.dtype!=float or df.High.dtype!=float or df.Low.dtype!=float or df.Close.dtype!=float or (df.Volume.dtype!=int and df.Volume.dtype!=float)):
	df = pd.read_csv(datalocation, names = ['Time (EET)', 'EndTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
	if (df.Open.dtype!=float or df.High.dtype!=float or df.Low.dtype!=float or df.Close.dtype!=float or (df.Volume.dtype!=int and df.Volume.dtype!=float)):
		print ("\n is not in the correct data format of ['Time (EET)', 'EndTime', 'Open', 'High', 'Low', 'Close', 'Volume']")
		print ("\n Please use a data csv file with the correct format")
		exit(-2)

# Take off Open, High, Low, Close and Volume from the Dataframe to input_data                
input_data = df[['Time (EET)', 'EndTime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Data Definition - must be given by the user
indexes = df.index
indexes = indexes[:300].values.reshape(-1,1)
input_data = input_data['Close'][:300]

# Plot Financial Data
plt.plot(indexes, input_data)

# Calling Example
LinearityDetector(50, 70)
LinearityDetector(10, 40)
LinearityDetector(120, 190)

plt.show()

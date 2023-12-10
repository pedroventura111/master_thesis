# ============================================================================================= #
# =========================================== NOTES =========================================== #
# ============================================================================================= #

"""
This script represents the Market Classification Module
"""

# ============================================================================================= #
# ========================================= LIBRARIES ========================================= #
# ============================================================================================= #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import pickle

# My own written code
from machine_learning_methods import LogisticRegression, SVM, RandomForest, XGBoost, ClassifierEvaluation
from sklearn import preprocessing

# Define plot's size
plt.rcParams['figure.figsize'] = (10, 8)

# Exclude unnecessary error 
pd.options.mode.chained_assignment = None  # default='warn'


# ============================================================================================= #
# ======================================== DEFINITIONS ======================================== #
# ============================================================================================= #

# Trigger to classify the ensemble voting
# ensemble_trigger = 3.5

# Technical Indicators to print together with the data
print1_ti = 'BB_UP' 	#red
print2_ti = 'BB_MIDDLE'	#green
print3_ti = 'BB_LOW'	#orange

# Number of values to ignore due to NaN values
ignore_nan = 300

# File to be written when saving environment 
file_to_save = 'Renko10-20year-200'

# ============================================================================================= #
# =========================================== PLOTS =========================================== #
# ============================================================================================= #

# Plot all the predictions at the same time
def PlotAllPredictions(pred1, pred2, pred3, pred4, true):
	
	fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1, 1, 1, 1, 1]})
	ax1.set_title("Prediction")
	
	ax1.plot(indexes_test, input_data_test, 'blue', linewidth=0.5)
	ax1.plot(technical_indicators_test[print1_ti], 'red', linewidth=0.5)
	ax1.plot(technical_indicators_test[print2_ti], 'green', linewidth=0.5)
	ax1.plot(technical_indicators_test[print3_ti], 'orange', linewidth=0.5)
	ax1.set_ylabel('Forex Data', color='blue')
	ax1.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax1.grid(True)
	
	ax2.plot(true, color='green')
	ax2.set_ylabel('True', color='green')
	ax2.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax2.grid(True)

	ax3.plot(pred1, color='red')
	ax3.set_ylabel('Log Reg', color='orange')
	ax3.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax3.grid(True)

	ax4.plot(pred2, color='red')
	ax4.set_ylabel('SVM', color='orange')
	ax4.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax4.grid(True)

	ax5.plot(pred3, color='red')
	ax5.set_ylabel('Rand For', color='orange')
	ax5.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax5.grid(True)

	ax6.plot(pred4, color='red')
	ax6.set_ylabel('XGB', color='orange')
	ax6.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax6.grid(True)

# Compares the classification with the respective prediction
def PlotPrediction(prediction, true, title):

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1, 1]})
	ax1.set_title(title)
	
	ax1.plot(indexes_test, input_data_test, 'blue', linewidth=0.5)
	ax1.plot(technical_indicators_test[print1_ti], 'red', linewidth=0.5)
	ax1.plot(technical_indicators_test[print2_ti], 'green', linewidth=0.5)
	ax1.plot(technical_indicators_test[print3_ti], 'orange', linewidth=0.5)
	ax1.set_ylabel('Forex Data', color='blue')
	ax1.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax1.grid(True)
	
	ax2.plot(prediction, color='green')
	ax2.set_ylabel('Prediction', color='green')
	ax2.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax2.grid(True)

	ax3.plot(true, color='orange')
	ax3.set_ylabel('True', color='orange')
	ax3.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax3.grid(True)

# Compares the pre and pos processed data
def PlotProcessment(prediction, true, title):

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1, 1]})
	ax1.set_title(title)
	
	ax1.plot(indexes_test, input_data_test, 'blue', linewidth=0.5)
	ax1.plot(technical_indicators_test[print1_ti], 'red', linewidth=0.5)
	ax1.plot(technical_indicators_test[print2_ti], 'green', linewidth=0.5)
	ax1.plot(technical_indicators_test[print3_ti], 'orange', linewidth=0.5)
	ax1.set_ylabel('Forex Data', color='blue')
	ax1.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax1.grid(True)
	
	ax2.plot(prediction, color='green')
	ax2.set_ylabel('Prediction', color='green')
	ax2.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax2.grid(True)

	ax3.plot(true, color='orange')
	ax3.set_ylabel('True', color='orange')
	ax3.xaxis.set_ticks(np.arange(0, len(indexes_test), 100))
	ax3.grid(True)


# ============================================================================================= #
# ========================================= FUNCTIONS ========================================= #
# ============================================================================================= #

# Min Max Normalization - Data
def MinMaxNormalization(min_max_scaler, data):

	# Search for the min and max values of the data
	min_max_scaler.fit(data.reshape(-1,1))
	aux_data = min_max_scaler.transform(data.reshape(-1,1))
	data = aux_data.reshape(len(aux_data))
	
	# Normalizes the initial data
	#for ii in range(len(data)):
	#	aux_data = min_max_scaler.transform(data[ii].reshape(-1,1))
	#	data[ii] = aux_data.reshape(len(aux_data))

	return data
	
# Performes min/max normalization
def DataNormalization(data):

	# Normalization type
	min_max_scaler = preprocessing.MinMaxScaler()

	# Input data normalization
	for ii in range(len(data)):
		data[ii] = MinMaxNormalization(min_max_scaler, data[ii])

	return np.transpose(data)

# Calculates the max/min/mean of the last 10/20/30 canclesticks
def MaxMinMean(data, n_candlesticks):

	maximum = np.zeros(len(data))
	minimum = np.zeros(len(data))
	mean = np.ones(len(data))

	for ii in range(n_candlesticks, len(data)):
		maximum[ii] = max(data[ii - n_candlesticks : ii])
		minimum[ii] = min(data[ii - n_candlesticks : ii])
		mean[ii] = np.mean(data[ii - n_candlesticks : ii])

	return maximum, minimum, mean

# Calculates (max - min), (max / mean), (min / mean) and (atual_price / mean) in the last n_candlesticks
def DiverseFeatures0(data, n_candlesticks, xx_train):

	max, min, mean = MaxMinMean(data, n_candlesticks)
	max_min = max - min
	weighted_max = np.divide(max, mean)
	weighted_min = np.divide(min, mean)
	weighted_price = np.divide(data, mean)

	feature1 = np.reshape(np.array(max_min), (-1, 1))
	feature2 = np.reshape(np.array(weighted_max), (-1, 1))
	feature3 = np.reshape(np.array(weighted_min), (-1, 1))
	feature4 = np.reshape(np.array(weighted_price), (-1, 1))

	return np.concatenate([xx_train, feature1, feature2, feature3, feature4], axis=1)

# Calculates (max - min), (max), (min) and (mean) in the last n_candlesticks
def DiverseFeatures1(data, n_candlesticks, xx_train):

	max, min, mean = MaxMinMean(data, n_candlesticks)
	max_min = max - min

	feature1 = np.reshape(np.array(max_min), (-1, 1))
	feature2 = np.reshape(np.array(max), (-1, 1))
	feature3 = np.reshape(np.array(min), (-1, 1))
	feature4 = np.reshape(np.array(mean), (-1, 1))

	return np.concatenate([xx_train, feature1, feature2, feature3, feature4], axis=1)

# Calculates (max - min), (max - data), (min - data) and (mean - data) in the last n_candlesticks
def DiverseFeatures2(data, n_candlesticks, xx_train):

	max, min, mean = MaxMinMean(data, n_candlesticks)
	max_min = max - min
	max_data = max - data
	min_data = min - data
	mean_data = mean - data

	feature1 = np.reshape(np.array(max_min), (-1, 1))
	feature2 = np.reshape(np.array(max_data), (-1, 1))
	feature3 = np.reshape(np.array(min_data), (-1, 1))
	feature4 = np.reshape(np.array(mean_data), (-1, 1))

	return np.concatenate([xx_train, feature1, feature2, feature3, feature4], axis=1)

# Give the percentage of the data that was classified as sideways
def PercentageOfSideways(data):
	
	percentage = data.count(1) * 100 / len(data)
	percentage = "{:.3f}".format(percentage)
	print('In', len(data), 'elements,', percentage, '% is Sideways') 

# Calculates the difference between ii and ii-step
def ValuesDifference0(data, start, stop, xx_train, step):

	yy = np.zeros(len(data))

	for ii in range(start, stop):
		yy[ii - start + step] = data[ii] - data[ii - step]

	yy = np.reshape(np.array(yy), (-1, 1))

	return np.concatenate([xx_train, yy], axis=1)

# Returns the value on the index ii
def ValuesDifference1(data, start, stop, xx_train, step):

	yy = np.zeros(len(data))

	for ii in range(start, stop):
		yy[ii - start + step] = data[ii] 

	yy = np.reshape(np.array(yy), (-1, 1))

	return np.concatenate([xx_train, yy], axis=1)

# Pos-processment of the data
def DataPosProcessement(arr1, arr2, arr3, arr4, ensemble_trigger):
    
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)    
    arr4 = np.array(arr4)    
    result = arr1 + arr2 + arr3 + arr4
    
    classification = [0 if val < ensemble_trigger else 1 for val in result]
    #result = [0 if val < ensemble_trigger else val for val in result]

    return classification  # Convert back to a regular Python list

# Load variables to avoid time spending runs
def LoadEnvironment():

	# Load variables from the file
	with open('/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/SavedEnvironments/' + file_to_save + '-0.pkl', 'rb') as file:
		loaded_data = pickle.load(file)

	type_data = loaded_data['type_data']
	input_data = loaded_data['input_data']
	training_test_division = loaded_data['training_test_division']
	indexes_train = loaded_data['indexes_train']
	indexes_test = loaded_data['indexes_test']
	input_data_train = loaded_data['input_data_train']
	input_data_test = loaded_data['input_data_test']
	technical_indicators_train = loaded_data['technical_indicators_train']
	technical_indicators_test = loaded_data['technical_indicators_test']
	yy_train = loaded_data['yy_train']
	yy_test = loaded_data['yy_test']

	return type_data, input_data, training_test_division, indexes_train, indexes_test, input_data_train, input_data_test, technical_indicators_train, technical_indicators_test, yy_train, yy_test

# Save variables to avoid time spending runs
def SaveEnvironment():

	data_to_save = {
	    'final_prediction0_5': final_prediction0_5,
	    'final_prediction1_5': final_prediction1_5,
	    'final_prediction2_5': final_prediction2_5,
	    'final_prediction3_5': final_prediction3_5,
	    'log_reg_prediction': log_reg_prediction, 
        'svm_prediction': svm_prediction,
	    'rand_for_prediction': rand_for_prediction, 
        'xgb_prediction': xgb_prediction, 
    	'true': rand_for_true,
	    'xx_train': xx_train,
	    'xx_test': xx_test,
	    'log_reg_fit': log_reg_fit,
	    'svm_fit': svm_fit,
	    'rand_for_fit': rand_for_fit,
	    'xgb_fit': xgb_fit
	}

	# Save variables to a file
	with open('/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/SavedEnvironments/' + file_var_name + '-1.pkl', 'wb') as file:
		pickle.dump(data_to_save, file)

# Decides which DiverseFeatures() to use
def DiverseFeatures(data, n_candlesticks, xx_train, div_feat):
	if div_feat == 0:
		return DiverseFeatures0(data, n_candlesticks, xx_train)
	elif div_feat == 1:
		return DiverseFeatures1(data, n_candlesticks, xx_train)
	else:
		return DiverseFeatures2(data, n_candlesticks, xx_train)
	
# Decides which ValuesDifference() to use
def ValuesDifference(data, start, stop, xx_train, step, val_diff):
	if val_diff == 0:
		return ValuesDifference0(data, start, stop, xx_train, step)
	else:
		return ValuesDifference1(data, start, stop, xx_train, step)

# Defines both xx_train and xx_test
def ChooseXxTrainTest(all_features, val_diff, div_feat):

	# Train Features
	xx_train = np.reshape(np.array(input_data_train), (-1, 1)) 														# (0) Price 
	xx_train = ValuesDifference(input_data_train, 1, training_test_division, xx_train, 1, val_diff)							# (1) Price's difference: x[i] - x[i-1]

	if all_features == 1:
		xx_train = ValuesDifference(technical_indicators_train['SMA30'], 1, training_test_division, xx_train, 1, val_diff)		# (2) SMA30's difference
		xx_train = ValuesDifference(technical_indicators_train['SMA50'], 1, training_test_division, xx_train, 1, val_diff)		# (3) SMA50's difference
		xx_train = ValuesDifference(technical_indicators_train['BB_UP'], 1, training_test_division, xx_train, 1, val_diff)		# (5) BB_UP's difference
		xx_train = ValuesDifference(technical_indicators_train['BB_MIDDLE'], 1, training_test_division, xx_train, 1, val_diff)	# (6) BB_MIDDLE's difference
		xx_train = ValuesDifference(technical_indicators_train['BB_LOW'], 1, training_test_division, xx_train, 1, val_diff)		# (7) BB_LOW's difference
		xx_train = ValuesDifference(technical_indicators_train['RSI'], 1, training_test_division, xx_train, 1, val_diff)			# (8) RSI's difference

	xx_train = ValuesDifference(technical_indicators_train['SMA100'], 1, training_test_division, xx_train, 1, val_diff)		# (4) SMA100's difference
	xx_train = ValuesDifference(technical_indicators_train['SMA200'], 1, training_test_division, xx_train, 1, val_diff)		# (4) SMA200's difference
	xx_train = ValuesDifference(technical_indicators_train['SMA300'], 1, training_test_division, xx_train, 1, val_diff)		# (4) SMA300's difference
	#xx_train = DiverseFeatures(input_data_train, 10, xx_train, div_feat)														# (09)(10)(11)(12) (max-min), (max/mean), (min/mean) and (price/mean) in the last 10 candlesticks
	#xx_train = DiverseFeatures(input_data_train, 20, xx_train, div_feat)														# (13)(14)(15)(16) (max-min), (max/mean), (min/mean) and (price/mean) in the last 20 candlesticks
	#xx_train = DiverseFeatures(input_data_train, 30, xx_train, div_feat)														# (17)(18)(19)(20) (max-min), (max/mean), (min/mean) and (price/mean) in the last 30 candlesticks
	xx_train = DiverseFeatures(input_data_train, 50, xx_train, div_feat)														# (21)(22)(23)(24) (max-min), (max/mean), (min/mean) and (price/mean) in the last 50 candlesticks
	xx_train = DiverseFeatures(input_data_train, 100, xx_train, div_feat)														# (25)(26)(27)(28) (max-min), (max/mean), (min/mean) and (price/mean) in the last 100 candlesticks
	xx_train = DiverseFeatures(input_data_train, 200, xx_train, div_feat)														# (25)(26)(27)(28) (max-min), (max/mean), (min/mean) and (price/mean) in the last 100 candlesticks

	# Test Features
	xx_test = np.reshape(np.array(input_data_test), (-1, 1)) 																			# Price 
	xx_test = ValuesDifference(input_data_test, training_test_division + 1, len(input_data_test), xx_test, 1, val_diff)						# Price's difference: x[i] - x[i-1]

	if all_features == 1:
		xx_test = ValuesDifference(technical_indicators_test['SMA30'], 1, len(technical_indicators_test['SMA30']), xx_test, 1, val_diff)				# sma30's difference
		xx_test = ValuesDifference(technical_indicators_test['SMA50'], 1, len(technical_indicators_test['SMA50']), xx_test, 1, val_diff)				# SMA50's difference
		xx_test = ValuesDifference(technical_indicators_test['BB_UP'], 1, len(technical_indicators_test['BB_UP']), xx_test, 1, val_diff)				# BB_UP's difference
		xx_test = ValuesDifference(technical_indicators_test['BB_MIDDLE'], 1, len(technical_indicators_test['BB_MIDDLE']), xx_test, 1, val_diff)		# BB_MIDDLE's difference
		xx_test = ValuesDifference(technical_indicators_test['BB_LOW'], 1, len(technical_indicators_test['BB_LOW']), xx_test, 1, val_diff)			# BB_LOW's difference
		xx_test = ValuesDifference(technical_indicators_test['RSI'], 1, len(technical_indicators_test['RSI']), xx_test, 1, val_diff)					# RSI's difference

	xx_test = ValuesDifference(technical_indicators_test['SMA100'], 1, len(technical_indicators_test['SMA100']), xx_test, 1, val_diff)			# SMA100's difference
	xx_test = ValuesDifference(technical_indicators_test['SMA200'], 1, len(technical_indicators_test['SMA200']), xx_test, 1, val_diff)			# SMA200's difference
	xx_test = ValuesDifference(technical_indicators_test['SMA300'], 1, len(technical_indicators_test['SMA300']), xx_test, 1, val_diff)			# SMA300's difference
	#xx_test = DiverseFeatures(input_data_test, 10, xx_test, div_feat)																				# (max-min), (max/mean), (min/mean) and (price/mean) in the last 10 candlesticks
	#xx_test = DiverseFeatures(input_data_test, 20, xx_test, div_feat)																				# (max-min), (max/mean), (min/mean) and (price/mean) in the last 20 candlesticks
	#xx_test = DiverseFeatures(input_data_test, 30, xx_test, div_feat)																				# (max-min), (max/mean), (min/mean) and (price/mean) in the last 30 candlesticks
	xx_test = DiverseFeatures(input_data_test, 50, xx_test, div_feat)																				# (max-min), (max/mean), (min/mean) and (price/mean) in the last 50 candlesticks	
	xx_test = DiverseFeatures(input_data_test, 100, xx_test, div_feat)																			# (max-min), (max/mean), (min/mean) and (price/mean) in the last 100 candlesticks	
	xx_test = DiverseFeatures(input_data_test, 200, xx_test, div_feat)																			# (max-min), (max/mean), (min/mean) and (price/mean) in the last 100 candlesticks	

	xx_train = DataNormalization(np.transpose(xx_train))
	xx_test = DataNormalization(np.transpose(xx_test))

	return xx_train, xx_test

# ============================================================================================= #
# =================================== FEATURES DEFINITION ===================================== #
# ============================================================================================= #

print('########################################################')

# Access command-line arguments
file_var_name = sys.argv[1]
all_features0 = int(sys.argv[2])
val_diff0 = int(sys.argv[3])
div_feat0 = int(sys.argv[4])
all_features1 = int(sys.argv[5])
val_diff1 = int(sys.argv[6])
div_feat1 = int(sys.argv[7])
all_features2 = int(sys.argv[8])
val_diff2 = int(sys.argv[9])
div_feat2 = int(sys.argv[10])
all_features3 = int(sys.argv[11])
val_diff3 = int(sys.argv[12])
div_feat3 = int(sys.argv[13])

# Load Data
type_data, input_data, training_test_division, indexes_train, indexes_test, input_data_train, input_data_test, technical_indicators_train, technical_indicators_test, yy_train, yy_test = LoadEnvironment()
PercentageOfSideways(yy_train)
PercentageOfSideways(yy_test)

# ============================================================================================= #
# ================================= MACHINE LEARNING METHODS ================================== #
# ============================================================================================= #

""" Performes a Logistic Regression with binary classification {Sideways Market; Non-Sideways Market} """
xx_train, xx_test = ChooseXxTrainTest(all_features0, val_diff0, div_feat0)
log_reg_prediction, log_reg_true, log_reg_fit = LogisticRegression(xx_train[:][ignore_nan:], xx_test[:][ignore_nan:], yy_train[:][ignore_nan:], yy_test[:][ignore_nan:], ignore_nan)
#PlotPrediction(log_reg_prediction, log_reg_true, "Logistic Regression")

"""Performes a Suport Vector Machine with binary classification {Sideways Market; Non-Sideways Market} """
xx_train, xx_test = ChooseXxTrainTest(all_features1, val_diff1, div_feat1)
#svm_prediction = np.zeros(len(log_reg_prediction))
svm_prediction, svm_true, svm_fit = SVM(xx_train[:][ignore_nan:], xx_test[:][ignore_nan:], yy_train[:][ignore_nan:], yy_test[:][ignore_nan:], ignore_nan)
#PlotPrediction(svm_prediction, svm_true, "Suport Vector Machine")

""" Performes a Random Forest with binary classification {Sideways Market; Non-Sideways Market} """
xx_train, xx_test = ChooseXxTrainTest(all_features2, val_diff2, div_feat2)
rand_for_prediction, rand_for_true, rand_for_fit = RandomForest(xx_train[:][ignore_nan:], xx_test[:][ignore_nan:], yy_train[:][ignore_nan:], yy_test[:][ignore_nan:], ignore_nan)
#PlotPrediction(rand_for_prediction, rand_for_true, "Random Forest")

""" Performes a XGBoost with binary classification {Sideways Market; Non-Sideways Market} """
xx_train, xx_test = ChooseXxTrainTest(all_features3, val_diff3, div_feat3)
xgb_prediction, xgb_true, xgb_fit = XGBoost(xx_train[:][ignore_nan:], xx_test[:][ignore_nan:], yy_train[:][ignore_nan:], yy_test[:][ignore_nan:], ignore_nan)
#PlotPrediction(xgb_prediction, xgb_true, "XGBoost")

final_prediction0_5 = DataPosProcessement(log_reg_prediction, rand_for_prediction, xgb_prediction, svm_prediction, 0.5)
#PlotProcessment(final_prediction2, rand_for_true, "All together 3.5")
#ClassifierEvaluation(rand_for_true, final_prediction0_5)

final_prediction1_5 = DataPosProcessement(log_reg_prediction, rand_for_prediction, xgb_prediction, svm_prediction, 1.5)
#PlotProcessment(final_prediction2, rand_for_true, "All together 3.5")
#ClassifierEvaluation(rand_for_true, final_prediction1_5)

#PlotAllPredictions(log_reg_prediction, svm_prediction, rand_for_prediction, xgb_prediction, xgb_true)
final_prediction2_5 = DataPosProcessement(log_reg_prediction, rand_for_prediction, xgb_prediction, svm_prediction, 2.5)
#PlotProcessment(final_prediction1, rand_for_true, "All together 2.5")
#ClassifierEvaluation(rand_for_true, final_prediction2_5)

final_prediction3_5 = DataPosProcessement(log_reg_prediction, rand_for_prediction, xgb_prediction, svm_prediction, 3.5)
#ClassifierEvaluation(rand_for_true, final_prediction3_5)

SaveEnvironment() 
# ============================================================================================= #
# =========================================== NOTES =========================================== #
# ============================================================================================= #

"""
This script classifies the train set in sections of sideways and non-sideways markets - Data Processing Module
"""

# ============================================================================================= #
# ========================================= LIBRARIES ========================================= #
# ============================================================================================= #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from matplotlib.lines import Line2D
from talib import SMA, EMA, BBANDS, SAR, ADX, CCI, MFI, MOM, PPO, ROC, STOCH, WILLR, ATR, AD, MACD, RSI, OBV
from sklearn import preprocessing

# My own written code
from machine_learning_methods import LinearRegression

# Define plot's size
plt.rcParams['figure.figsize'] = (10, 8)

# Exclude unnecessary error 
pd.options.mode.chained_assignment = None  # default='warn'


# ============================================================================================= #
# ======================================== DEFINITIONS ======================================== #
# ============================================================================================= #

# Training/Test rate
training_percentage = 0.8

# Data type used {Open, Close, High, Low, Volume}
type_data = 'Close' 

# Technical Indicators to print together with the data
print1_ti = 'BB_UP' 	#red
print2_ti = 'BB_MIDDLE'	#green
print3_ti = 'BB_LOW'	#orange

# Step to search for the next horizontal space
step = 10

# Size of the sideways market to search for
size = 200 

# Slope max to a linear regression be admited as sideways market
slope_max = 5e-5

# Minimum score of the sideways market (before expansion): 10000 * (up_limit - low_limit) / (right_limit - left_limit)
initialscore = 2

# Boundarie to avoid on 1st/last values because 1st/last values are much bigger than others
boundarie = 30

# Margin given to a up/low limit (if value + error < limit, continue)
error = 0.004

# Minimum score of the sideways market: 10000 * (up_limit - low_limit) / (right_limit - left_limit)
minscore = 0.6 

# Slope max to a rejected score be considered sideways market
#low_slope = 1e-5
low_slope = 0

# Minimum size of an sideways section
sideways_min_xx = 200

# Number max of candlesticks that we can ignore to concatenate 2 sideways 
sideways_max_dist = 50

# To define the begin_threshold we compute a Linear Regression with a certain slope's threshold: the initial lr size is defined as size * initial_lr 
initial_lr = 0.1

# To define the begin_threshold we compute a Linear Regression with a certain slope's threshold: is the increasing step size
step_lr = 10

# To define the begin_threshold we compute a Linear Regression with a certain slope's threshold: is the slope's threshold
slope_max_lr = slope_max

# File localization
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Candlestick_1_Hour_BID_04.05.2003-14.04.2023.csv'  # Hour - Bigger Set
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Hourly_Ask_2022.01.01_2022.12.31.csv' 			    # Hour - Smaller Set
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_5_PIPS_Ticks_Bid_2005.01.01_2023.06.06.csv'  # Renko 5 - Bigger set
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_5_PIPS_Ticks_Bid_2005.01.01_2009.12.31.csv'  # Renko 5 - 5 year set
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_10_PIPS_Ticks_Bid_2022.01.01_2022.12.31.csv' # Renko 10 - 1 year set
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_10_PIPS_Ticks_Bid_2021.01.01_2022.12.31.csv' # Renko 10 - 2 year set
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_10_PIPS_Ticks_Bid_2018.01.01_2022.12.31.csv' # Renko 10 - 5 year set
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_10_PIPS_Ticks_Bid_2005.01.01_2009.12.31.csv'  # Renko 10 - 5 year set
datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_10_PIPS_Ticks_Bid_2005.01.01_2023.06.06.csv'  # Renko 10 - 18 year set
#datalocation = r'/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/Data/EURUSD_Renko_10_PIPS_Ticks_Bid_2003.05.05_2020.12.31.csv'  # Renko 10 - 20 year set (new)

# File to be written when saving environment 
file_to_save = 'Renko10-20year-200-newnew'


# ============================================================================================= #
# =========================================== PLOTS =========================================== #
# ============================================================================================= #

# Plot a Section
def PlotSection(m, xx, yy):

	if abs(m) < slope_max:
		plt.plot(xx, yy, 'black',  linewidth = 2.5) 
	else:
		plt.plot(xx, yy, 'red',  linewidth = 2.5) 

# Defines the legend, ticks, title, etc of a Plot
def PlotShow(data):
	"""
	legend_lines = [Line2D([0], [0], color = 'black', lw = 4),
            Line2D([0], [0], color = 'grey', lw = 4),
            Line2D([0], [0], color = 'red', lw = 4),
            Line2D([0], [0], color = 'orange', lw = 4),
            Line2D([0], [0], color = 'yellow', lw = 4),
			Line2D([0], [0], color = 'green', lw = 4)]
	ax1.legend(legend_lines, ['<< 60h', '<< 120h', '<< 180h', '<< 240h', '<< 300h', '>> 300h'])
	"""

	legend_lines = [Line2D([0], [0], color = 'blue', lw = 4),
			Line2D([0], [0], color = 'black', lw = 4),
			Line2D([0], [0], color = 'red', lw = 4),
	]
	plt.legend(legend_lines, ['Financial Data', 'Accepted Slope', 'Declined Slope'])

	plt.ylabel('Currency Pair Quote', color='black')
	#ax1.set_ylabel('Currency Pair Quote', color='black')
	plt.xlabel('Sample Number', color='black')	
	#ax1.set_xticks(np.arange(0, len(data), 100))
	plt.xticks(np.arange(0, len(data), 100))
	#ax1.grid(True)


# ============================================================================================= #
# ========================================= FUNCTIONS ========================================= #
# ============================================================================================= #

# Give the percentage of the data that was classified as sideways
def PercentageOfSideways(data):
	
	percentage = data.count(1) * 100 / len(data)
	percentage = "{:.3f}".format(percentage)
	print('In', len(data), 'elements,', percentage, '% is Sideways') 

# Choose a color to plot: the color depend on the duration of the sideways market
def ChooseColor(length):

	half_week = 60 # 5 buniness days * 24h * 1/2
	
	if length < half_week: 		 # 0 -- 1/2 weeks
		return 'black'
	elif length < 2 * half_week: # 1/2 -- 1 weeks
		return 'grey'
	elif length < 3 * half_week: # 1 -- 3/2 weeks
		return 'red'
	elif length < 4 * half_week: # 3/2 -- 2 weeks
		return 'orange'
	elif length < 5 * half_week: # 2 -- 5/2 weeks
		return 'yellow'
	else:						 # >> 5/2 weeks
		return 'green'

# Remove the weekends from the initial dataset: 48h weekends (friday 5pm - sunday 5pm) + 120h business days
def RemoveWeekends(input_data):

	weekend_counter = 21 # Sunday 00am - 9pm (+ 1)
	weekend_indexes = [] # Indexes that are weekend and will be removed
	flag = True 		 # Indicate if is the 1st hour to remove
	ii = 0

	while ii < len(input_data[type_data]):
		if(flag):
			#print("Elimina de:", input_data['Date'][ii], ii)
			#plt.vlines(x = ii, ymin = 0.8, ymax = 1.2, color = 'red')
			flag = False
		if(weekend_counter != 0):
			weekend_indexes.append(ii)
			weekend_counter -= 1
		else:
			#print("Até_______:", input_data['Date'][ii-1], ii-1)
			#plt.vlines(x = ii-1, ymin = 0.8, ymax = 1.2, color = 'orange')
			flag = True
			ii += 120
			weekend_counter = 47
		ii += 1

	# Remove weekends from initial dataset
	input_data = input_data.drop(weekend_indexes)
	input_data = input_data.reset_index(drop=True)

	return input_data

# Find the Min and Max of the found section and try to expand the section (compensates error)
def ExpandSection(xx, yy, index, slope, to_plot, yy_pred):
	
	# Support and Resistance Level
	up_limit = max(yy[index + boundarie: index + size - boundarie])
	low_limit = min(yy[index + boundarie: index + size - boundarie])

	# Compute the Initial Score
	score = (up_limit - low_limit)*10000/(size - boundarie * 2)
	if score > initialscore:
		return 0, index, False, 0, 0

	# Right Low Expansion
	first = True
	aux = len(xx) - index - size + boundarie
	for ii, vv in enumerate(yy[index + size - boundarie:]):
		if (vv < low_limit - error) & first:
			aux = ii - 1
			break
		elif (vv < low_limit) & first: 
			aux = ii - 1 
			first = False
		elif vv > low_limit:
			first = True
			aux = len(xx) - index - size + boundarie
		elif vv < low_limit - error:
			break
	right_low_limit = aux + index + size - boundarie

	# Right Up Expansion
	first = True
	aux = len(xx) - index - size
	for ii, vv in enumerate(yy[index + size - boundarie:]):
		if (vv > up_limit + error) & first:
			aux = ii - 1
			break
		elif (vv > up_limit) & first: 
			aux = ii - 1 
			first = False
		elif vv < up_limit:
			first = True
			aux = len(xx) - index - size + boundarie
		elif vv > up_limit + error:
			break
	right_up_limit = aux + index + size - boundarie

	# Right Boundary
	right_limit = min(right_low_limit, right_up_limit)
	
	# Left Low Expansion
	first = True
	aux = 0
	for ii, vv in reversed(list(enumerate(yy[:index + boundarie]))):
		if (vv < low_limit - error) & first:
			aux = ii + 1
			break
		elif (vv < low_limit) & first: 
			aux = ii + 1 
			first = False
		elif vv > low_limit:
			first = True
			aux = 0
		elif vv < low_limit - error:
			break
	left_low_limit = aux

	# Left Up Expansion
	first = True
	aux = 0
	for ii, vv in reversed(list(enumerate(yy[:index + boundarie]))):
		if (vv > up_limit + error) & first:
			aux = ii + 1
			break
		elif (vv > up_limit) & first: 
			aux = ii + 1 
			first = False
		elif vv < up_limit:
			first = True
			aux = 0
		elif vv > up_limit + error:
			break
	left_up_limit = aux

	# Left Boundary
	left_limit = max(left_low_limit, left_up_limit)

	# Compute the Final Score
	score = (up_limit - low_limit)*10000/(right_limit - left_limit)
	sideways = False	
	if score > minscore:
		# Return Non-Sideways
		if abs(slope) > low_slope:
			return 0, index, sideways, 0, 0
		#else:
		#	plt.plot(xx[index: index + size - 1], yy_pred, 'pink',  linewidth = 2.5) 
	elif right_limit - left_limit < sideways_min_xx:
		# Return Non-Sideways
		return 0, index, sideways, 0, 0
	else:
		# Return Sideways 
		sideways = True

	# Plot the Sideways Section
	if (to_plot):
		# Plot the sideways' x_limits
		ax1.vlines(x = index + boundarie, ymin = low_limit, ymax = up_limit, color = 'grey')
		ax1.vlines(x = index + size - 1 - boundarie, ymin = low_limit, ymax = up_limit, color = 'grey')

		# Plot the limits where the max/min where searched
		#plt.vlines(x = index + boundarie, ymin = low_limit, ymax = up_limit, color = 'grey')
		#plt.vlines(x = index + size - boundarie, ymin = low_limit, ymax = up_limit, color = 'grey')

		#xx_length = ChooseColor(right_limit - left_limit)
		xx_length = 'green'
		# Plot the up/down limit
		ax1.hlines(y = up_limit, xmin = left_limit, xmax = right_limit, color = xx_length)
		ax1.hlines(y = low_limit, xmin = left_limit, xmax = right_limit, color = xx_length)

		# Plot the up/down limit +/- error
		ax1.hlines(y = up_limit + error, xmin = left_limit, xmax = right_limit, color = 'yellow')
		ax1.hlines(y = low_limit - error, xmin = left_limit, xmax = right_limit, color = 'yellow')
	
	#PlotSection(slope, xx[index: index + size - 1], yy_pred)

	return left_limit, right_limit, sideways, up_limit, low_limit

# Decides the best sideways if 2 of them have a common space
def CompareSideways(previous_left, previous_right, left, right, yy_final, previous_min, previous_max, minimum, maximum):

	score = (maximum - minimum) * 10000 / (right - left)
	previous_score = (previous_max - previous_min) * 10000 / (previous_right - previous_left)

	# If the new sideways has a socre worst than the previous, ignore the new sideways and continue the search
	if previous_score < score:
		print('Eliminou posterior:', left, right)
		return False, yy_final
	# If the new sideways is grater than the previous, delete the previous and set the new as sideways
	else:
		print('Eliminou anterior:', previous_left, previous_right)
		yy_final[previous_left : previous_right] = [0] * (previous_right - previous_left) 
		return True, yy_final

# Calculates begin_threshold - can't classify sideways perfectly because it isn't possible to the algorithm to understand it so quickly
def ComputeBeginThreshold(xx, yy, left, right):

	begin = left + boundarie
	begin_threshold = round((right - left) * initial_lr)
	m = 1000
	to_plot = True

	# Compute Linear Dectection until the desired slope
	while abs(m) > slope_max_lr:
		
		begin_threshold += step_lr
		if begin + begin_threshold > right:
			print('Erro:', left, right)
			begin_threshold = 150
			to_plot = False
			break
		
		___, m, yy_pred = LinearRegression(xx[begin : begin + begin_threshold], yy[begin : begin + begin_threshold])
		#ax1.plot(xx[begin: begin + begin_threshold], yy_pred, 'pink',  linewidth = 2.5)

	
	#if to_plot:
		#ax1.plot(xx[begin: begin + begin_threshold], yy_pred, 'orange',  linewidth = 2.5)

	return begin_threshold + boundarie

# Find all the subsections with Linear Regression Slope < slope_max
def ZeroSlopeSections(xx, yy, technical_indicators, to_plot):
	
	#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
	plt.figure()

	# Initializations
	yy_final = [0 for ___ in range(len(xx))]
	index = 0
	previous_left = 0
	previous_right = 0
	first = True

	# Compute until the end of the data set
	while (index < len(yy) - size):

		# Performes a Linear Regression returning the score and the slope
		___, m, yy_pred = LinearRegression(xx[index: index + size - 1], yy[index: index + size - 1])
		
		PlotSection(m, xx[index: index + size - 1], yy_pred)

		# Accepted Slope
		if (abs(m) < slope_max):
			# Expand Section
			left_limit, index, sideways, maximum, minimum = ExpandSection(xx, yy, index, m, to_plot, yy_pred)
			if (sideways) or (abs(m) <= low_slope):
				# Compute the initial threshold
				begin_threshold = ComputeBeginThreshold(xx, yy, left_limit, index)
				define_as_sideways = True
				concatenated = False
				if (left_limit - previous_right <= sideways_max_dist) and (not first):
					# Concatenate the two sideways in one bigger
					if(((previous_max + error >= maximum) and (previous_min - error <= minimum)) or ((maximum + error >= previous_max) and (minimum - error <= previous_min))):
						#print('Concatenou:', previous_right, left_limit + begin_threshold)													
						yy_final[previous_right : left_limit + begin_threshold] = [1] * (left_limit + begin_threshold - previous_right)		
						concatenated = True
						#if (previous_right < left_limit):
							#print('Concatenou', previous_right)
				# If there are a space in common w/ the previous sideways
				if (not concatenated) and (previous_right > left_limit + begin_threshold):
					define_as_sideways, yy_final = CompareSideways(previous_left, previous_right, left_limit, index, yy_final, previous_min, previous_max, minimum, maximum) 
				# If the new one is better than the previous
				if (define_as_sideways):
					yy_final[left_limit + begin_threshold: index] = [1] * (index - left_limit - begin_threshold) 
					if concatenated:
						previous_left = previous_left
						previous_right = index
						previous_max = max(previous_max, maximum)
						previous_min = min(previous_min, minimum)
					else:
						previous_left = left_limit + begin_threshold
						previous_right = index
						previous_max = maximum
						previous_min = minimum
						first = False
					#print('Sideways:', left_limit + begin_threshold, index)
				else:
					index += step
			else:
				#plt.plot(xx[index: index + size - 1], yy_pred, 'orange',  linewidth = 2.5) 
				index += step
		else:
			index += step	

	if(to_plot):	
		ax1.plot(xx, yy, 'blue', linewidth=0.5)
		#ax1.plot(technical_indicators[print1_ti], 'red', linewidth=0.5)
		#ax1.plot(technical_indicators[print2_ti], 'green', linewidth=0.5)
		#ax1.plot(technical_indicators[print3_ti], 'orange', linewidth=0.5)
	
		PlotShow(xx, ax1)
	
		ax2.plot(yy_final, color='green')
		ax2.set_ylabel('Classification', color='black')
		#ax2.set_xlabel('Sample Number', color='green')


	plt.plot(xx, yy, 'blue', linewidth=0.5)
	PlotShow(xx)


	return yy_final

# Min Max Normalization - Technical Indicators
def MinMaxNormalizationTI(ti_name, min_max_scaler, data):

	# Search for the min and max values of the training data and normalizes it in the range [-1,1]
	aux_data = min_max_scaler.fit_transform(data[ti_name].values.reshape(-1,1))
	data[ti_name] = aux_data.reshape(len(aux_data))

# Min Max Normalization - Data
def MinMaxNormalization(min_max_scaler, data, technical_indicators):

	# Search for the min and max values of the data
	min_max_scaler.fit(data.values.reshape(-1,1))

	# Normalizes the initial data
	aux_data = min_max_scaler.transform(data.values.reshape(-1,1))
	data = aux_data.reshape(len(aux_data))

	"""Normalizes some technical indicators applying the scaller fitted with the input_data"""
	# BB_UP
	aux_data = min_max_scaler.transform(technical_indicators['BB_UP'].values.reshape(-1,1))
	technical_indicators['BB_UP'] = aux_data.reshape(len(aux_data))
	# BB_MIDDLE
	aux_data = min_max_scaler.transform(technical_indicators['BB_MIDDLE'].values.reshape(-1,1))
	technical_indicators['BB_MIDDLE'] = aux_data.reshape(len(aux_data))
	# BB_LOW
	aux_data = min_max_scaler.transform(technical_indicators['BB_LOW'].values.reshape(-1,1))
	technical_indicators['BB_LOW'] = aux_data.reshape(len(aux_data))
	# SMA30
	aux_data = min_max_scaler.transform(technical_indicators['SMA30'].values.reshape(-1,1))
	technical_indicators['SMA30'] = aux_data.reshape(len(aux_data))
	# SMA50
	aux_data = min_max_scaler.transform(technical_indicators['SMA50'].values.reshape(-1,1))
	technical_indicators['SMA50'] = aux_data.reshape(len(aux_data))
	# SMA100
	aux_data = min_max_scaler.transform(technical_indicators['SMA100'].values.reshape(-1,1))
	technical_indicators['SMA100'] = aux_data.reshape(len(aux_data))
	# SMA200
	aux_data = min_max_scaler.transform(technical_indicators['SMA200'].values.reshape(-1,1))
	technical_indicators['SMA200'] = aux_data.reshape(len(aux_data))
	# SMA300
	aux_data = min_max_scaler.transform(technical_indicators['SMA300'].values.reshape(-1,1))
	technical_indicators['SMA300'] = aux_data.reshape(len(aux_data))

	return data
	
# Performes min/max normalization on technical indicators
def TechnicalIndicatorsNormalization(technical_indicators, min_max_scaler):

	#MinMaxNormalizationTI('SAR', min_max_scaler, technical_indicators)
	MinMaxNormalizationTI('ADX', min_max_scaler, technical_indicators)
	MinMaxNormalizationTI('CCI', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('MFI', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('MOM', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('PPO', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('ROC', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('STOCH_K', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('STOCH_D', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('WILL_R', min_max_scaler, technical_indicators)
	MinMaxNormalizationTI('ATR', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('AD', min_max_scaler, technical_indicators)

	#MinMaxNormalizationTI('MACD_Line', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('MACD_Signal', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('MACD_Hist', min_max_scaler, technical_indicators)
	MinMaxNormalizationTI('RSI', min_max_scaler, technical_indicators)
	#MinMaxNormalizationTI('OBV', min_max_scaler, technical_indicators)

# Performes min/max normalization
def DataNormalization(data, technical_indicators):

	# Normalization type
	min_max_scaler = preprocessing.MinMaxScaler()

	# Input data normalization
	data = MinMaxNormalization(min_max_scaler, data, technical_indicators)

	TechnicalIndicatorsNormalization(technical_indicators, min_max_scaler)

	return data

# Compute Technical Indicators - Uses talib over pd.ta because of the speed
def TechnicalIndicators(data, begin, end):

	adx = np.zeros(end-begin)
	cci = np.zeros(end-begin)
	rsi = np.zeros(end-begin)
	atr = np.zeros(end-begin)

	sma3 = SMA(data.values, timeperiod=3)
	sma5 = SMA(data.values, timeperiod=5)
	sma10 = SMA(data.values, timeperiod=10)

	# Use talib to obtain technical indicators
	sma30 = SMA(data.values, timeperiod=30)
	sma50 = SMA(data.values, timeperiod=50)
	sma100 = SMA(data.values, timeperiod=100)
	sma200 = SMA(data.values, timeperiod=200)
	sma300 = SMA(data.values, timeperiod=300)
	
	bb_upp, bb_middle, bb_lower = BBANDS(data.values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
	#sar = SAR(input_data['High'].values, input_data['Low'].values, acceleration=0.02, maximum=0.2)
	adx[0:end-begin] = ADX(input_data['High'][begin:end].values, input_data['Low'][begin:end].values, input_data['Close'][begin:end].values, timeperiod=14)
	cci[0:end-begin] = CCI(input_data['High'][begin:end].values, input_data['Low'][begin:end].values, input_data['Close'][begin:end].values, timeperiod=20)
	#mfi = MFI(input_data['High'].values, input_data['Low'].values, input_data['Close'].values, input_data['Volume'].values, timeperiod=14)
	#mom = MOM(input_data[type_data].values, timeperiod=10)
	#ppo = PPO(input_data[type_data].values, fastperiod=12, slowperiod=26, matype=0)
	#roc = ROC(input_data[type_data].values, timeperiod=10)
	#stoch_k, stoch_d = STOCH(input_data['High'].values, input_data['Low'].values, input_data['Close'].values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
	#willr = WILLR(input_data['High'].values, input_data['Low'].values, input_data['Close'].values, timeperiod=14)
	atr[0:end-begin] = ATR(input_data['High'][begin:end].values, input_data['Low'][begin:end].values, input_data['Close'][begin:end].values, timeperiod=14)
	#ad = AD(input_data['High'].values, input_data['Low'].values, input_data['Close'].values, input_data['Volume'].values)
	
	#macd = MACD(input_data[type_data].values) #default fastperiod=12, slowperiod=26, signalperiod=9 https://sourceforge.net/p/ta-lib/code/HEAD/tree/branches/ta-lib/ta-lib/c/src/ta_func/ta_MACD.c#l124
	#macd_line = macd[0]
	#macd_signal = macd[1]
	#macd_hist = macd[2]
	rsi = RSI(data.values) #default timeperiod=14 https://sourceforge.net/p/ta-lib/code/HEAD/tree/branches/ta-lib/ta-lib/c/src/ta_func/ta_RSI.c
	#obv = OBV(input_data[type_data].values, input_data['Volume'].values)
	
	aux_sma3 = pd.DataFrame(sma3, columns = ['SMA3'])
	aux_sma5 = pd.DataFrame(sma5, columns = ['SMA5'])
	aux_sma10 = pd.DataFrame(sma10, columns = ['SMA10'])

	# Stores the tecnhical indicators in a dataframe
	aux_sma30 = pd.DataFrame(sma30, columns = ['SMA30'])
	aux_sma50 = pd.DataFrame(sma50, columns = ['SMA50'])
	aux_sma100 = pd.DataFrame(sma100, columns = ['SMA100'])
	aux_sma200 = pd.DataFrame(sma200, columns = ['SMA200'])
	aux_sma300 = pd.DataFrame(sma300, columns = ['SMA300'])
	
	#aux_ema30 = pd.DataFrame(ema30, columns = ['EMA30'])
	#aux_ema50 = pd.DataFrame(ema50, columns = ['EMA50'])
	#aux_ema100 = pd.DataFrame(ema100, columns = ['EMA100'])
	#aux_ema200 = pd.DataFrame(ema200, columns = ['EMA200'])
	#aux_ema300 = pd.DataFrame(ema300, columns = ['EMA300'])

	aux_bb_upp = pd.DataFrame(bb_upp, columns = ['BB_UP'])
	aux_bb_middle = pd.DataFrame(bb_middle, columns = ['BB_MIDDLE'])
	aux_bb_lower = pd.DataFrame(bb_lower, columns = ['BB_LOW'])
	#aux_sar = pd.DataFrame(sar, columns = ['SAR'])
	aux_adx = pd.DataFrame(adx, columns = ['ADX'])
	aux_cci = pd.DataFrame(cci, columns = ['CCI'])
	#aux_mfi = pd.DataFrame(mfi, columns = ['MFI'])
	#aux_mom = pd.DataFrame(mom, columns = ['MOM'])
	#aux_ppo = pd.DataFrame(ppo, columns = ['PPO'])
	#aux_roc = pd.DataFrame(roc, columns = ['ROC'])
	#aux_stoch_k = pd.DataFrame(stoch_k, columns = ['STOCH_K'])
	#aux_stoch_d = pd.DataFrame(stoch_d, columns = ['STOCH_D'])
	#aux_willr = pd.DataFrame(willr, columns = ['WILL_R'])
	aux_atr = pd.DataFrame(atr, columns = ['ATR'])
	#aux_ad = pd.DataFrame(ad, columns = ['AD'])

	#aux_macd_line = pd.DataFrame(macd_line, columns = ['MACD_Line'])
	#aux_macd_signal = pd.DataFrame(macd_signal, columns = ['MACD_Signal'])
	#aux_macd_hist = pd.DataFrame(macd_hist, columns = ['MACD_Hist'])
	aux_rsi = pd.DataFrame(rsi, columns = ['RSI'])
	#aux_obv = pd.DataFrame(obv, columns = ['OBV'])
	
	# Stores all the technical indicators in the same variable 
	technical_indicators = pd.concat([aux_sma3, aux_sma5, aux_sma10, aux_sma30, aux_sma50, aux_sma100, aux_sma200, aux_sma300, aux_bb_upp, aux_bb_middle, aux_bb_lower, aux_adx, aux_cci, aux_atr, aux_rsi], axis=1)

	return technical_indicators
	
# Save variables to avoid time spending runs
def SaveEnvironment():

	data_to_save = {
		'type_data': type_data,
    	'input_data': input_data,
	    'training_test_division': training_test_division,
	    'indexes_train': indexes_train,
	    'indexes_test': indexes_test,
	    'input_data_train': input_data_train,
	    'input_data_test': input_data_test,
	    'technical_indicators_train': technical_indicators_train,
	    'technical_indicators_test': technical_indicators_test,
	    'yy_train': yy_train,
	    'yy_test': yy_test,
	}

	# Save variables to a file
	with open('/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/SavedEnvironments/' + file_to_save + '-0.pkl', 'wb') as file:
		pickle.dump(data_to_save, file)


# ============================================================================================= #
# =========================================== MAIN ============================================ #
# ============================================================================================= #

# Dataframe with data from the csv file
df = pd.read_csv(datalocation, names = ['Time (EET)', 'EndTime', 'Open', 'High', 'Low', 'Close', 'Volume'], skiprows = 1)
#df = pd.read_csv(datalocation, names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], skiprows = 1)

# Verifies missing values in the csv file and deletes the corresponding row of the csv file
if (df.Open.dtype!=float or df.High.dtype!=float or df.Low.dtype!=float or df.Close.dtype!=float or (df.Volume.dtype!=int and df.Volume.dtype!=float)):
	df = pd.read_csv(datalocation, names = ['Time (EET)', 'EndTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
	#df = pd.read_csv(datalocation, names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
	if (df.Open.dtype!=float or df.High.dtype!=float or df.Low.dtype!=float or df.Close.dtype!=float or (df.Volume.dtype!=int and df.Volume.dtype!=float)):
		print ("\n is not in the correct data format of ['Time (EET)', 'EndTime', 'Open', 'High', 'Low', 'Close', 'Volume']")
		print ("\n Please use a data csv file with the correct format")
		exit(-2)

# Take off Open, High, Low, Close and Volume from the Dataframe to input_data                
input_data = df[['Time (EET)', 'EndTime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
#input_data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Remove weekends from the initial data
#input_data = RemoveWeekends(input_data)

# Finds the limit between the training and test data in the input data
training_test_division = int(len(input_data[type_data]) * training_percentage)

# Take off the indexes to an array
indexes = df.index
indexes = indexes.values.reshape(-1,1)
# Because I removed the weekends from the initial data
indexes = indexes[: len(input_data[type_data])]

# Divide in train/test xx
indexes_train = indexes[:training_test_division]
indexes_test = indexes[:(len(indexes) - training_test_division)] # Indexes starting in 0
indexes_test_true = indexes[training_test_division:]			  # Indexes starting in training_test_division

# Divides the data in xx_train and xx_test
input_data_train = input_data[type_data][:training_test_division]
input_data_test = input_data[type_data][training_test_division:]

# Calculate the technical indicators
technical_indicators_train = TechnicalIndicators(input_data_train, 0 , training_test_division)
technical_indicators_test = TechnicalIndicators(input_data_test, training_test_division, len(input_data['Close']))

# Find all the subsections with low Linear Regression's Slope
yy_train = ZeroSlopeSections(indexes_train, input_data_train, technical_indicators_train, False)
yy_test = ZeroSlopeSections(indexes_test, input_data_test, technical_indicators_test, False)

# Performs Data Normalization in the range [0, 1]
#input_data_train = DataNormalization(input_data_train, technical_indicators_train)
#input_data_test = DataNormalization(input_data_test, technical_indicators_test)

SaveEnvironment() 
#PercentageOfSideways(yy_train)
#PercentageOfSideways(yy_test)
plt.show()

# Needed computations for the report
"""
entry=0
exit=0
counter=0
for ii in range(1, len(yy_train)):
	if yy_train[ii-1] == 0 and yy_train[ii] == 1:
		entry+=1
	elif yy_train[ii-1] == 1 and yy_train[ii] == 0:
		exit+=1
	if yy_train[ii] == 1:
		counter+=1

print(entry, exit, counter)
"""
# ============================================================================================= #
# ============================================ NOTES ========================================== #
# ============================================================================================= #

"""
This script computes the trading framework - Final Prediction Module & Trading Framework Module
"""


# ============================================================================================= #
# ========================================== LIBRARIES ======================================== #
# ============================================================================================= #

import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib.lines import Line2D
from machine_learning_methods import ClassifierEvaluation

# Define plot's size
plt.rcParams['figure.figsize'] = (10, 10)

# ============================================================================================= #
# ======================================== DEFINITIONS ======================================== #
# ============================================================================================= #

# Initial money to invest 100,000€
initial_money = 100000

# Eventual fees/commissions to pay during a trade
single_trade_fee = 0.0001

# Every 1,000,000 US$ spent, the broker charges 20 US$
accumulated_trade_max = 1e6 
accumulated_trade_fee = 0
accumulated_trade_fee = 20

# Actual money to invest
actual_money = initial_money 

# Number minimum of votes to execute a buy/sell opportunity
min_votes = 3

# Resistance and Support level - xx range considering that the algorithm has a delay until finding a sideways market
res_sup_range = 100

# To avoid great losses
stop_gain = 0.010
stop_loss = 0.015

# Technical Indicators boundaries to consider as a buy/sell opportunity
rsi_max = 65
rsi_min = 35
cci_max = 165
cci_min = -165

# File to be written when saving environment
file_to_load0 = 'Renko10-20year-200'
file_to_load1 = '20y-200-star'    

# ============================================================================================= #
# ====================================== OTHER FUNCTIONS ====================================== #
# ============================================================================================= #

# Load variables from process_data.py
def LoadEnvironment_0():

	# Load variables from the file
	with open('/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/SavedEnvironments/' + file_to_load0 + '-0.pkl', 'rb') as file:
		loaded_data = pickle.load(file)

	training_test_division = loaded_data['training_test_division']
	indexes_test = loaded_data['indexes_test']
	input_data_test = loaded_data['input_data_test']
	technical_indicators_test = loaded_data['technical_indicators_test']

	return training_test_division, indexes_test, input_data_test, technical_indicators_test

# Load variables from market_classification.py
def LoadEnvironment_1():
    
	# Load variables from the file
	with open('/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/SavedEnvironments/' + file_to_load1 + '-1.pkl', 'rb') as file:
		loaded_data = pickle.load(file)
                
	final_prediction0_5 = loaded_data['final_prediction0_5']
	final_prediction1_5 = loaded_data['final_prediction1_5']
	final_prediction2_5 = loaded_data['final_prediction2_5']
	final_prediction3_5 = loaded_data['final_prediction3_5']
	true = loaded_data['true']
	log_reg_prediction = loaded_data['log_reg_prediction']    
	svm_prediction = loaded_data['svm_prediction']    
	rand_for_prediction = loaded_data['rand_for_prediction']    
	xgb_prediction = loaded_data['xgb_prediction']    

	#ClassifierEvaluation(true, log_reg_prediction)    
	#ClassifierEvaluation(true, svm_prediction)    
	#ClassifierEvaluation(true, rand_for_prediction)    
	#ClassifierEvaluation(true, xgb_prediction)    

	return final_prediction0_5, final_prediction1_5, final_prediction2_5, final_prediction3_5, true, log_reg_prediction

# Transform an vector with indexes in an array of '0' with '1' in the indexes
def ConstructVector(indexes):
     
    aux = np.zeros(len(data))
    for ii, val in enumerate(indexes):
        aux[val] = 1

    return aux

# Defines the legend, ticks, title, etc of a Plot
def PlotShow(data, ax1):

	legend_lines = [Line2D([0], [0], color = 'blue', lw = 4),]
	ax1.legend(legend_lines, ['Financial Data'])

	#plt.ylabel('Currency Pair Quote', color='black')
	ax1.set_ylabel('Currency Pair Quote', color='black')
	plt.xlabel('Sample Number', color='black')	
	ax1.set_xticks(np.arange(0, len(data), 100))
	#plt.xticks(np.arange(0, len(data), 100))
	ax1.grid(True)

# Plot the hypoteses and the final sideways' classification
def PlotPrediction(final_classification):

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1, 1]})
	ax1.set_title('Final Classification')
        
	ax1.plot(indexes, data, 'blue', linewidth=0.5)
	#ax1.plot(technical_indicators['BB_UP'], 'red', linewidth=0.5)
	#ax1.plot(technical_indicators['BB_MIDDLE'], 'green', linewidth=0.5)
	#ax1.plot(technical_indicators['BB_LOW'], 'red', linewidth=0.5)
	ax1.set_ylabel('Currency Pair Quote', color='black')
	ax1.xaxis.set_ticks(np.arange(0, len(data), 100))
	ax1.grid(True)      

	ax2.plot(final_classification, 'orange')
	ax2.set_ylabel('Prediction', color='black')
	ax2.xaxis.set_ticks(np.arange(0, len(final_classification), 100))
	ax2.grid(True)

	ax3.plot(true, 'green')
	ax3.set_xlabel("Samples Number", color='black')
	ax3.set_ylabel('True', color='black')
	ax3.xaxis.set_ticks(np.arange(0, len(true), 100))
	ax3.grid(True) 

# Plot all the buying and selling opportunities
def PlotOpportunities():

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1, 1]})
	ax1.set_title('Buy & Sell Opportunities')
        
	ax1.plot(indexes, data, 'blue', linewidth=0.5)
	ax1.plot(technical_indicators['BB_UP'], 'red', linewidth=0.5)
	ax1.plot(technical_indicators['BB_MIDDLE'], 'red', linewidth=0.5)
	ax1.plot(technical_indicators['BB_LOW'], 'red', linewidth=0.5)
	ax1.plot(technical_indicators['SMA3'], 'green', linewidth=0.5)
	ax1.plot(technical_indicators['SMA10'], 'orange', linewidth=0.5)

#	ax1.scatter(final_buying_indexes, data[[val + training_test_division for val in final_buying_indexes]], color='black', s=50)
#	ax1.scatter(final_selling_indexes, data[[val + training_test_division for val in final_selling_indexes]], color='black', s=50)

	#ax1.scatter(max_delay_buying_indexes, data[[val + training_test_division for val in max_delay_buying_indexes]], color='green', s=100)
	#ax1.scatter(min_delay_selling_indexes, data[[val + training_test_division for val in min_delay_selling_indexes]], color='green', s=100)

	ax1.scatter(rsi_buying_indexes, data[[val + training_test_division for val in rsi_buying_indexes]], color='red', s=100)
	ax1.scatter(rsi_selling_indexes, data[[val + training_test_division for val in rsi_selling_indexes]], color='red', s=100)
	ax1.scatter(cci_buying_indexes, data[[val + training_test_division for val in cci_buying_indexes]], color='green', s=50)
	ax1.scatter(cci_selling_indexes, data[[val + training_test_division for val in cci_selling_indexes]], color='green', s=50)

	ax1.scatter(bb_buying_indexes, data[[val + training_test_division for val in bb_buying_indexes]], s=30, color="black", marker="x")
	ax1.scatter(bb_selling_indexes, data[[val + training_test_division for val in bb_selling_indexes]], color='black', s=30, marker="x")

	#ax1.scatter(max_buying_indexes, data[[val + training_test_division for val in max_buying_indexes]], color='black', s=10)
	#ax1.scatter(min_selling_indexes, data[[val + training_test_division for val in min_selling_indexes]], color='black', s=10)
      
	legend_lines = [Line2D([0], [0], color = 'black', lw = 4),
            Line2D([0], [0], color = 'red', lw = 4),
            Line2D([0], [0], color = 'green', lw = 4),
            Line2D([0], [0], color = 'green', lw = 4)]

	ax1.legend(legend_lines, ['BB', 'RSI', 'CCI', 'Delay'])
	ax1.set_ylabel('Finantial Data', color='blue')
	ax1.xaxis.set_ticks(np.arange(0, len(data), 100))
	ax1.grid(True)      
        
	ax2.plot(final_classification, 'orange')
	ax2.set_ylabel('Sideways', color='orange')
	ax2.xaxis.set_ticks(np.arange(0, len(data), 100))
	ax2.grid(True)      

	ax3.plot(true, 'green')
	ax3.set_ylabel('True', color='green')
	ax3.xaxis.set_ticks(np.arange(0, len(data), 100))
	ax3.grid(True)      
        
# Plot the final trade
def PlotTrade():

	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
	ax1.set_title('Final Trade')
        
	ax1.plot(indexes, data, 'blue', linewidth=0.5)
	#ax1.plot(technical_indicators['BB_UP'], 'red', linewidth=0.5)
	#ax1.plot(technical_indicators['BB_MIDDLE'], 'red', linewidth=0.5)
	#ax1.plot(technical_indicators['BB_LOW'], 'red', linewidth=0.5)
	#ax1.plot(technical_indicators['SMA3'], 'green', linewidth=0.5)
	#ax1.plot(technical_indicators['SMA10'], 'orange', linewidth=0.5)

	ax1.scatter(final_buying_indexes, data[[val + training_test_division for val in final_buying_indexes]], color='green', s=35)
	ax1.scatter(final_selling_indexes, data[[val + training_test_division for val in final_selling_indexes]], color='red', s=35)

	ax1.set_ylabel('Finantial Data', color='black')
	ax1.xaxis.set_ticks(np.arange(0, len(data), 100))
	ax1.grid(True)      
        
	ax2.plot(final_classification, 'orange')
	ax2.set_ylabel('Prediction', color='black')
	ax2.xaxis.set_ticks(np.arange(0, len(data), 100))
	ax2.grid(True)      
        
	legend_lines = [Line2D([0], [0], color = 'blue', lw = 4),
			Line2D([0], [0], color = 'green', lw = 4),
			Line2D([0], [0], color = 'red', lw = 4),
	]
	ax1.legend(legend_lines, ['Currency Pair Price', 'Short Position', 'Long Position'])
	plt.xlabel('Sample Number', color='black')	

	"""
    ax3.plot(true, 'green')
	ax3.set_ylabel('Classification', color='green')
	ax3.xaxis.set_ticks(np.arange(0, len(data), 100))
	ax3.grid(True)
    """

# Specific plot to the report
def AuxiliarPlot(data_to_plot):

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1, 1]})
	ax1.set_title('Final Prediction')
        
	ax1.plot(indexes, data, 'blue', linewidth=0.5)
	ax1.set_ylabel('Finantial Data', color='blue')
	ax1.xaxis.set_ticks(np.arange(0, len(data), 100))
	ax1.grid(True)      

	ax2.plot(data_to_plot, 'orange')
	ax2.set_ylabel('Prediction', color='orange')
	ax2.xaxis.set_ticks(np.arange(0, len(data_to_plot), 100))
	ax2.grid(True)

	ax3.plot(true, 'green')
	ax3.set_ylabel('True Classification', color='green')
	ax3.set_xlabel("Sample's Number", color='green')
	ax3.xaxis.set_ticks(np.arange(0, len(true), 100))
	ax3.grid(True) 

# Plots the evolution of the ROI
def RoiPerTradePlot(num_trades):

    plt.figure()
    plt.plot(roi_per_trade)
    
    plt.grid(True)    
    plt.xlabel('Trade Number')
    plt.ylabel('Actual ROI [%]')
    plt.title('Overall ROI Evolution')

# ============================================================================================= #
# ========================================== TRADING ========================================== #
# ============================================================================================= #

# Calculates the DrawDown and the RunUp
def DrawDown_RunUp(num_trades):
    
    # Roi per Trade will increase or decrease
    up_down = list()
    # Indexes where occurs inversio
    local_max = list()
    local_min = list() 
    inversions = list()

    # Local max/min minus Local min/max
    minus = list()
    
    for ii in range(1, len(roi_per_trade)):
        if roi_per_trade[ii] > roi_per_trade[ii - 1]:
            up_down.append(+1)
        else: 
            up_down.append(-1)

    for ii in range(len(up_down)):
        if ((up_down[ii - 1] == 1) and (up_down[ii] == -1)):
            local_max.append(ii)
        
        if ((up_down[ii - 1] == -1) and (up_down[ii] == 1)):
            local_min.append(ii)

    # Concatenate lists
    inversions = local_max + local_min
    inversions.append(0)
    inversions.append(len(roi_per_trade) - 1)
    # Sort the concatenated list
    inversions = sorted(inversions)

    for ii in range(1, len(inversions)):
        minus.append(roi_per_trade[inversions[ii]] - roi_per_trade[inversions[ii-1]])
   
    run_up = max(minus)
    draw_down = min(minus)

    print('DrawDown = ', draw_down, '%')
    print('RunUp = ', run_up, '%')

# Computes the profit
def Profit(actual_money, num_trades, num_positive_trades, num_negative_trades):
    
    roi = (actual_money - initial_money) * 100 / initial_money

    print('ROI = ', roi, '%')
    print('Profit = ', roi/num_trades, '%')
    print('Positive Trades = ', num_positive_trades * 100 / num_trades, '%')
    print('Negative Trades = ', num_negative_trades * 100 / num_trades, '%')
    print('Trades = ', num_trades)

# Looks for a Long Position opportunity
def LookForLongOpportunity(ii, actual_money, lower_than_buy, last_buy, cross_bb_middle, previous_eur, num_positive_trades, num_negative_trades):
     
    # Initializations
    dollars_spent = 0
    buying = False
    step = 0
    num_votes = bb_selling[ii] + rsi_selling[ii] + cci_selling[ii] #+ min_delay_selling[ii] 
    
    # Trading Rules
    rule0 = bb_selling[ii] == 1
    rule1 = num_votes >= min_votes
    rule2 = min_delay_selling[ii] == 1
    rule_stop_gain = data[ii + training_test_division] <= data[last_buy] - stop_gain
    rule_stop_loss = data[ii + training_test_division] >= data[last_buy] + stop_loss
    rule_control = lower_than_buy and (data[ii + training_test_division] >= data[last_buy] - single_trade_fee) and (cross_bb_middle)
    rule_bb_middle = (data[ii + training_test_division] >= technical_indicators['BB_MIDDLE'][ii]) and (cross_bb_middle)

    # The used rules found an long position opportunity
    if rule1 or rule_stop_gain:    
        final_selling_indexes.append(ii)

        # Compute the exchange
        dollars_spent = actual_money
        actual_money = actual_money / (data[ii + training_test_division] + single_trade_fee)
        
        # Computes some evaluation metrics
        roi_per_trade.append((actual_money - initial_money) * 100 / initial_money)
        num_positive_trades, num_negative_trades = ClassifyTrade(actual_money, previous_eur, num_positive_trades, num_negative_trades)
        buying = True
        cross_bb_middle = False
        lower_than_buy = False
        #print('⬇️:', ii, data[ii + training_test_division], "{:.3f}".format(actual_money), '€')

    return actual_money, buying, cross_bb_middle, dollars_spent, lower_than_buy, step, num_positive_trades, num_negative_trades

# Looks for a Short Position oportunity
def LookForShortOpportunity(ii, actual_money, num_trades, previous_eur):

    # Initializations
    num_votes = bb_buying[ii] + rsi_buying[ii] + cci_buying[ii] #+ max_delay_buying[ii] 
    buying = True
    first_after_buy = False

    # Trading Rules
    rule0 = bb_buying[ii] == 1
    rule1 = num_votes >= min_votes
    rule2 = max_delay_buying[ii] == 1
    rule3 = bb_buying[ii-2] + rsi_buying[ii-2] + cci_buying[ii-2] >=2 and bb_buying[ii-1] + rsi_buying[ii-1] + cci_buying[ii-1] >=2 and bb_buying[ii] + rsi_buying[ii] + cci_buying[ii] >=2

    # The used rules found an short position opportunity
    if rule1:
        final_buying_indexes.append(ii)

        # Compute the exchange
        previous_eur = actual_money
        actual_money = actual_money * (data[ii + training_test_division])

        # Computes some evaluation metrics
        num_trades += 1
        buying = False
        first_after_buy = True
        #print('⬆️:', ii, data[ii + training_test_division], "{:.3f}".format(actual_money), '$')

    return actual_money, buying, first_after_buy, num_trades, previous_eur

# Classify a trade as positive (profit) or negative (loss)
def ClassifyTrade(money_ii, money_ii_1, num_positive_trades, num_negative_trades):
     
    if money_ii >= money_ii_1:
        return num_positive_trades + 1, num_negative_trades
    else:
        return num_positive_trades, num_negative_trades + 1
    
# Trades during sideways markets
def TradingStrategy(actual_money):

    # Looking for a short position opportunity
    buying = True

    # Flags to control boughts in a local max during an growing market
    first_after_buy = False
    lower_than_buy = False
    cross_bb_middle = False
    last_buy = 0
    ii = 1
    previous_eur = actual_money
    num_trades = 0
    num_positive_trades = 0
    num_negative_trades = 0

    # Every 1.000.000 US$ spent, the broker charges 20 US$
    dollars_spent = 0

    while ii < len(final_classification):

        # After a short, if the market decreases, we need to control the possible losses
        if (first_after_buy):
            first_after_buy = False
            last_buy = ii - 1 + training_test_division

        # Sideways Market
        if final_classification[ii] == 1:
            # Short Option
            if buying:
                actual_money, buying, first_after_buy, num_trades, previous_eur = LookForShortOpportunity(ii, actual_money, num_trades, previous_eur)
            # Long Option
            else:
                if data[ii + training_test_division] < technical_indicators['BB_MIDDLE'][ii]:
                    cross_bb_middle = True
                if data[ii + training_test_division] < data[last_buy]:
                    lower_than_buy = True
                actual_money, buying, cross_bb_middle, spent, lower_than_buy, step, num_positive_trades, num_negative_trades = LookForLongOpportunity(ii, actual_money, lower_than_buy, last_buy, cross_bb_middle, previous_eur, num_positive_trades, num_negative_trades)   
                
                # Every 1e6 US$ spent, the broker charges 20 US$
                dollars_spent += spent
                if dollars_spent >= accumulated_trade_max:
                    actual_money = actual_money - accumulated_trade_fee / data[ii + training_test_division] 
                    #print('💵:', ii, "{:.3f}".format(actual_money), '€')
                    dollars_spent -= accumulated_trade_max
                ii += step
        # Switched from sideways to non-sideays - long
        elif final_classification[ii - 1] == 1:
            # We have to sell to avoid trending huge loss
            if (not buying):

                # Compute the trade
                actual_money = actual_money / (data[ii + training_test_division] + single_trade_fee)
                roi_per_trade.append((actual_money - initial_money) * 100 / initial_money)
                num_positive_trades, num_negative_trades = ClassifyTrade(actual_money, previous_eur, num_positive_trades, num_negative_trades)
                final_selling_indexes.append(ii)
            buying = True
            #print('⬇️:', ii, data[ii + training_test_division], "{:.3f}".format(actual_money), '€')
            #print('------------------------------------')
        ii += 1

    Profit(actual_money, num_trades, num_positive_trades, num_negative_trades)
    #PlotTrade()
    #RoiPerTradePlot(num_trades)
    DrawDown_RunUp(num_trades)

# Find all the points that: Touch the BB_UP/BB_LOW; Out of the RSI or CCI range; Touch the support or resistance levels
def FindAllPoints():

    for ii in range(len(final_classification[1:])):
        # Sideways Market
        if final_classification[ii] == 1:
            
            # BB influence
            if technical_indicators['BB_UP'][ii] <= data[ii + training_test_division]:      #if "{:.4f}".format(technical_indicators['BB_UP'][ii]) <= "{:.4f}".format(data[ii + training_test_division]):
                bb_buying_indexes.append(ii)           
            if technical_indicators['BB_LOW'][ii] >= data[ii + training_test_division]:     #if "{:.4f}".format(technical_indicators['BB_LOW'][ii]) >= "{:.4f}".format(data[ii + training_test_division]):    
                bb_selling_indexes.append(ii)

            # RSI influence
            if(technical_indicators['RSI'][ii] >= rsi_max):
                rsi_buying_indexes.append(ii)
            elif(technical_indicators['RSI'][ii] <= rsi_min):
                rsi_selling_indexes.append(ii)          

            # CCI influence
            if(technical_indicators['CCI'][ii] >= cci_max):
                cci_buying_indexes.append(ii)
            elif(technical_indicators['CCI'][ii] <= cci_min):
                cci_selling_indexes.append(ii)      

            # Sideways entry point 
            if final_classification[ii - 1] == 0:
                entry_point = ii -1

            # Resistance and Support Prices considering that the algorithm has a delay finding the entry point
            resistance_level_delay = max(data[max(0, entry_point - res_sup_range) : ii])
            support_level_delay = min(data[max(0, entry_point - res_sup_range) : ii])

            if (data[ii + training_test_division] >= resistance_level_delay):
                max_delay_buying_indexes.append(ii)
            elif (data[ii + training_test_division] <= support_level_delay):
                min_delay_selling_indexes.append(ii)

            # Resistance and Support Prices
            resistance_level = max(data[max(0, entry_point) : ii])
            support_level = min(data[max(0, entry_point) : ii]) 

            if (data[ii + training_test_division] >= resistance_level):
                max_buying_indexes.append(ii)
            elif (data[ii + training_test_division] <= support_level):
                min_selling_indexes.append(ii)

    #PlotOpportunities()

# Finds the entry and exit points --> '20y-200-star'    
def FindEntryExitPoints():
     
    final_classification = np.zeros(len(data))
    entry_exit_points = list()     
    
    # Indicates if we are in a sideways or looking for one
    sideways = False

    for ii in range(len(data)):

        # Looking for an entry point
        if (not sideways): 
            if (final_prediction2_5[ii] == 1):
                sideways = True
                entry_exit_points.append(ii)
        # Looking for an exit point
        else:
            if (final_prediction1_5[ii] == 0):
                sideways = False
                entry_exit_points.append(ii)

    ii = 0
    while 1:
        final_classification[entry_exit_points[ii] : entry_exit_points[ii+1]] = [1] * (entry_exit_points[ii+1] - entry_exit_points[ii])
        ii += 2
        if ii == len(entry_exit_points):
            break

    #PlotPrediction(final_classification)
    #AuxiliarPlot(final_classification)
    #AuxiliarPlot(log_reg_prediction)
    return final_classification

"""
# Finds the entry and exit points --> '20y-200-new-star'    
def FindEntryExitPoints():
     
    final_classification = np.zeros(len(data))
    entry_exit_points = list()     
    
    # Indicates if we are in a sideways or looking for one
    sideways = False

    for ii in range(len(data)):

        # Looking for an entry point
        if (not sideways): 
            if (final_prediction2_5[ii] == 1):
                sideways = True
                entry_exit_points.append(ii)
        # Looking for an exit point
        else:
            if (final_prediction0_5[ii] == 0 ) or (final_prediction1_5[ii] == 0) or (log_reg_prediction[ii] == 0):
                sideways = False
                entry_exit_points.append(ii)

    ii = 0
    while 1:
        final_classification[entry_exit_points[ii] : entry_exit_points[ii+1]] = [1] * (entry_exit_points[ii+1] - entry_exit_points[ii])
        ii += 2
        if ii == len(entry_exit_points):
            break

    PlotPrediction(final_classification)

    return final_classification       

"""

# ============================================================================================= #
# =========================================== MAIN ============================================ #
# ============================================================================================= #

# Load data set
training_test_division, indexes, data, technical_indicators = LoadEnvironment_0()

# Load predictions
final_prediction0_5, final_prediction1_5, final_prediction2_5, final_prediction3_5, true, log_reg_prediction = LoadEnvironment_1()

# Initializations
final_buying_indexes = list()
final_selling_indexes = list()
bb_buying_indexes = list()
bb_selling_indexes = list()
rsi_buying_indexes = list()
rsi_selling_indexes = list()
cci_buying_indexes = list()
cci_selling_indexes = list()
max_delay_buying_indexes = list()
min_delay_selling_indexes = list()
max_buying_indexes = list()
min_selling_indexes = list()
roi_per_trade = list()
roi_per_trade.append(0)

# Prediction Pos-processement
final_classification = FindEntryExitPoints()

# To measure the trading framework quality instead of the overall work quality 
#final_classification = true
#ClassifierEvaluation(true, final_classification)

# Finds all the possible buy/sell opportunities
FindAllPoints()

# Transform lists in vectors
bb_buying = ConstructVector(bb_buying_indexes)
bb_selling = ConstructVector(bb_selling_indexes)
rsi_buying = ConstructVector(rsi_buying_indexes)
rsi_selling = ConstructVector(rsi_selling_indexes)
cci_buying = ConstructVector(cci_buying_indexes)
cci_selling = ConstructVector(cci_selling_indexes)
max_delay_buying = ConstructVector(max_delay_buying_indexes)
min_delay_selling = ConstructVector(min_delay_selling_indexes)
max_buying = ConstructVector(max_buying_indexes)
min_selling = ConstructVector(min_selling_indexes)

# Apply Trading Strategy
TradingStrategy(actual_money)

plt.show()
"""Implemented but not used fuctions"""

# Concatenate 2 similar Sideways in 1: two in a row with the yy of one inside of the other and with a small xx_distance --> IS DOING SOMETHING WRONG BUT I'M NOT USING IT 
def ConcatenateSideways(data, sideways, to_plot):

	flag = False
	num_ones = 1
	one_indexes = [ii for ii, xx in enumerate(sideways) if xx == 1]

	for ii in range(1, len(one_indexes)):
		# Number of zeros in a row: diff
		diff = one_indexes[ii] - one_indexes[ii - 1]
		if(diff != 1 or (ii == len(one_indexes) - 1 and num_ones > 1)):
			if (flag):
				max2 = max(data[one_indexes[ii - num_ones] : (one_indexes[ii - 1] + 1)])
				min2 = min(data[one_indexes[ii - num_ones] : (one_indexes[ii - 1] + 1)])	
				# Check the compatability between max1/max2 and min1/min2 
				if(((max1 + error >= max2) and (min1 - error <= min2)) or ((max2 + error >= max1) and (min2 - error <= min1))):
					# Concatenate the 2nd sideways in 1st
					sideways[one_indexes[ii - num_ones - 1] : one_indexes[ii - num_ones]] = [1] * (one_indexes[ii - num_ones] - one_indexes[ii - num_ones - 1])
					print("Concatenou in", one_indexes[ii - num_ones], one_indexes[ii - num_ones - 1])
					#if(to_plot):
					#	plt.vlines(x = one_indexes[ii - num_ones - 1], ymin = 0.95, ymax = 1.25, color = 'green')
			if (diff <= sideways_max_dist + begin_threshold):
				max1 = max(data[one_indexes[ii - num_ones] : (one_indexes[ii - 1] + 1)])
				min1 = min(data[one_indexes[ii - num_ones] : (one_indexes[ii - 1] + 1)])
				flag = True
			else:
				flag = False
			num_ones = 1
		else:
			num_ones += 1

	return sideways

# Calculates the market type {0 - sideways; 1 - increasing; 2 - decreasing}
def IncreaseDecrease(data):
	
	# Initialize list  
	yy_train = list()

	# Set the 1st value of training as Null
	yy_train.append(0)
	# Math for training data
	for ii in range(1, training_test_division):
		if(data[type_data][ii] > data[type_data][ii-1]):
			yy_train.append(1)
		elif(data[type_data][ii] < data[type_data][ii-1]):
			yy_train.append(2)
		else:
			yy_train.append(0)

	# Initialize list  
	yy_test = list()

	# Set the 1st value of test as Null 
	yy_test.append(0)
	# Math for testing data
	for ii in range(training_test_division + 1, len(data[type_data])):
		if(data[type_data][ii] > data[type_data][ii-1]):
			yy_test.append(1)
		elif(data[type_data][ii] < data[type_data][ii-1]):
			yy_test.append(2)	
		else:
			yy_test.append(0)

	return yy_train, yy_test

# Trading Strategy
def oldTradingStrategy(actual_money):

    # Looking for a buy opportunity
    buying = True

    # Initializations
    final_buying_indexes = list()
    final_selling_indexes = list()

    for ii in range(len(final_prediction[1:])):

        # Sideways Market
        if final_prediction[ii] == 1:
            # Buying Option
            if buying:
                actual_money, final_buying_indexes, buying = LookForBuyingOpportunity(ii, actual_money, final_buying_indexes)
            # Selling Option
            else:
                actual_money, final_selling_indexes, buying = LookForSellingOpportunity(ii, actual_money, final_selling_indexes)        

        # Switched from sideways to non-sideays
        elif final_prediction[ii - 1] == 1:
            # We have to sell to avoid trending huge loss
            if (not buying):
                actual_money = (1 - fee) * actual_money / data[ii + training_test_division]
                final_selling_indexes.append(ii)
            buying = True
            print('⬇️:', ii, data[ii + training_test_division], "{:.3f}".format(actual_money), '€')
            print('------------------------------------')

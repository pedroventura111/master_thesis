# ============================================================================================= #
# =========================================== NOTES =========================================== #
# ============================================================================================= #

"""
This script contains the machine learning predictors
"""

# ============================================================================================= #
# ========================================= LIBRARIES ========================================= #
# ============================================================================================= #


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import graphviz
from matplotlib.lines import Line2D

# Machine Learning Methods
from sklearn.linear_model import LogisticRegression as LogRegr
from sklearn.linear_model import LinearRegression as LinRegr
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
from collections import Counter


# ============================================================================================= #
# ======================================== PROCESSEMENT ======================================= #
# ============================================================================================= #

# Count how many 1/0 in a row 
def MantainMovementInARow(data, begin):

	count = 0
	label = data[begin]

	for ii in range(begin, len(data)):
		if data[ii] == label:
			count += 1
		else: 
			break

	return count, begin, ii

# Process the Prediction
def ProcessPrediction(xgb_prediction):

    # Array with the 
    reps_in_a_row = [[],[]]
    end = 0
    
    while end < len(xgb_prediction):
        num_reps, begin, end = MantainMovementInARow(xgb_prediction, end)
        reps_in_a_row[0].append(num_reps)
        reps_in_a_row[1].append(begin)
        if (begin + num_reps == len(xgb_prediction)):
            break
        	
    return xgb_prediction

# Classication Evaluation
def ClassifierEvaluation(yy_true, yy_predict):

    target_names = ['0', '1']
    print(classification_report(yy_true, yy_predict, digits = 5, target_names = target_names))
    print(f"Test set predicted stats: {Counter(yy_predict)}")

# We have to ignore some values due to NaN values on technical indicators
def IgnoreNan(yy_true, ignore_nan, yy_pred, title):

    final_pred = np.zeros(len(yy_pred) + ignore_nan)
    final_pred[ignore_nan:] = yy_pred

    final_true = np.zeros(len(yy_true) + ignore_nan)
    final_true[ignore_nan:] = yy_true

    final_true = [aux + 0.0 for aux in final_true]
    final_pred = [aux + 0.0 for aux in final_pred]

    
    #plt.plot(final_true, 'green')
    #plt.plot(final_pred, 'orange')
    
    legend_lines = [Line2D([0], [0], color = 'green', lw = 4),
            Line2D([0], [0], color = 'orange', lw = 4),
			Line2D([0], [0], color = 'blue', lw = 4)]
    
    #plt.legend(legend_lines, ['True', 'Prediction', 'Data'])
    #plt.title(title)
    #plt.ylim(ymin = 0.95, ymax = 1.25)

    #plt.xticks(range(0, len(final_true), 100))
    #plt.grid(True)

    return final_pred, final_true


# ============================================================================================= #
# ========================================== METHODS ========================================== #
# ============================================================================================= #

# Logistic Regression with binary classification {1 - Sideways Market; 0 - Non-Sideways Market}
def LogisticRegression(xx, xx_test, yy, yy_true, ignore_nan):
    
    params0 = {"max_iter": 10000}

    params_hyper_opt = {'C': 2.584193694743781, 'fit_intercept': True, 'max_iter': 1000, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 0, 'solver': 'lbfgs', 'tol': 0.0005, 'warm_start': True}

    print('\n Training Logistic Regression...')
    logistic_regression = LogRegr(**params_hyper_opt).fit(xx, yy)
    yy_pred = logistic_regression.predict(xx_test)

    #train_score = logistic_regression.score(np.reshape(yy_true,(-1, 1)), yy_pred)    
    #print("Train Score = " + train_score.astype(str))

    #test_score = accuracy_score(yy_true, yy_pred)
    #print("Test Score = " + test_score.astype(str))

    ClassifierEvaluation(yy_true, yy_pred)

    final_pred, final_true = IgnoreNan(yy_true, ignore_nan, yy_pred, "Logistic Regression")
    
    return final_pred, final_true, logistic_regression

# Suport Vector Machine with binary classification {1 - Sideways Market; 0 - Non-Sideways Market}
def SVM(xx, xx_test, yy, yy_true, ignore_nan):
    
    params0 = {'kernel': 'poly'}

    print('\n Training Support Vector Machine...')
    svm = SVC(**params0).fit(xx, yy)
    yy_pred = svm.predict(xx_test)

    #score = svm.score(np.reshape(yy_true,(-1, 1)), yy_pred)    
    #print("Score = " + score.astype(str))

    ClassifierEvaluation(yy_true, yy_pred)

    final_pred, final_true = IgnoreNan(yy_true, ignore_nan, yy_pred, "Support Vector Machine")
    
    return final_pred, final_true, svm

# Random Forest with binary classification {1 - Sideways Market; 0 - Non-Sideways Market}
def RandomForest(xx, xx_test, yy, yy_true, ignore_nan):

    params0 = {}
    params_hyper_opt = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 8, 'max_features': 'log2', 'max_samples': 0.8300000000000001, 'min_samples_leaf': 5, 'n_estimators': 28, 'n_jobs': -1, 'random_state': 0}

    print('\n Training Random Forest Classifier...')
    random_forest = RandomForestClassifier(**params_hyper_opt).fit(xx, yy)
    yy_pred = random_forest.predict(xx_test)

    #score = random_forest.score(np.reshape(yy_true,(-1, 1)), yy_pred)    
    #print("Score = " + score.astype(str))

    #plt.plot(yy_true, 'green')
    #plt.plot(yy_pred, 'red') 

    ClassifierEvaluation(yy_true, yy_pred)

    final_pred, final_true = IgnoreNan(yy_true, ignore_nan, yy_pred, "Random Forest")
    
    return final_pred, final_true, random_forest

# XGBoost with binary classification {1 - Sideways Market; 0 - Non-Sideways Market}
def XGBoost(xx, xx_test, yy, yy_true, ignore_nan):

    params0 = {}
    params_hyper_opt = {'colsample_bytree': 0.9, 'learning_rate': 0.33962351233559196, 'max_depth': 22, 'min_child_weight': 4, 'n_estimators': 65, 'n_jobs': -1, 'random_state': 0, 'reg_alpha': 0.01, 'reg_lambda': 0.1, 'subsample': 0.8099999999999999}

    print('\n Training XGBoost...')
    yy = LabelEncoder().fit_transform(yy)
    xgb = XGBClassifier(**params_hyper_opt).fit(xx, yy)

    yy_pred = xgb.predict(xx_test)
    #print(classification_report(yy_true, yy_pred))

    #plot_importance(xgb)

    """
    confusion_matrix_ = confusion_matrix(yy_true, yy_pred)
    confusion_matrix_df = pd.DataFrame(confusion_matrix_, index=['0', '1'], columns=['0', '1'])
    
    plt.figure()
    sn.heatmap(confusion_matrix_df, annot=True, cmap='Greens', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    """

    ClassifierEvaluation(yy_true, yy_pred)

    #final_prediction = ProcessPrediction(yy_pred)

    final_pred, final_true = IgnoreNan(yy_true, ignore_nan, yy_pred, "XGBoost")
    
    return final_pred, final_true, xgb

# Linear Regression with binary classification
def LinearRegression(xx, yy):

    params = {'n_jobs': -1}

    linear_regression = LinRegr(**params).fit(xx, yy)
    yy_pred = linear_regression.predict(xx)

    m = linear_regression.coef_
    # b = linear_regression.intercept_

    score = linear_regression.score(xx, yy)    
    # print("Linear Regression Score = " + score.astype(str))

    mse = mean_squared_error(yy, yy_pred)
    # print("Linear Regression MSE = " + mse.astype(str))

    return mse, m, yy_pred

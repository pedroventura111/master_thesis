import numpy as np
import pickle

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll import scope

# Machine Learning Techniques
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression as LogRegr
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.fixes import loguniform

# Load variables from market_classification.py
def LoadEnvironment(file_to_load, load_type):
    
	# Load variables from the file
	with open('/Users/pedro/OneDrive - Universidade de Lisboa/Mestrado/5º Ano/Tese/Código/SavedEnvironments/' + file_to_load + '.pkl', 'rb') as file:
		loaded_data = pickle.load(file)
                
	xx = loaded_data[load_type]

	return xx[ignore_nan:]


# ============================================================================================= #
# ======================================== GRID SEARCH ======================================== #
# ============================================================================================= #

# Logistic Regression Tunning Parameters
def LogRegParameters():

    return {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'dual': [True, False],
        'tol': np.arange(0.00001, 0.001 + 0.0005, 0.0005).tolist(),
        'C': loguniform(0.01, 3),
        'fit_intercept': [True, False],
        'class_weight': ['dict', 'balanced', None],
        'random_state': [0], 
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        'max_iter': [1000],
        'multi_class': ['auto', 'ovr', 'multinomial'],
        #'n_jobs': [-1],
        'verbose': [0],
    }

    """
    return  {
        'warm_start' : hp.choice('warm_start', [True]),
        'tol' : hp.quniform('tol', 0.00001, 0.001, 0.0005),
        'fit_intercept' : hp.choice('fit_intercept', [False,True]),
        'C' : hp.uniform('C', 0.01, 3),
        'solver' : hp.choice('solver', ['lbfgs']),
        'penalty': hp.choice('penalty', ['l2']),
        'multi_class' : 'ovr',
        'random_state': [0],
        'max_iter': [1000],
        }
    """

# SVM Tunning Parameters
def SvmParameters():

    return {
        'C': [1, 5, 10, 15, 20],   # VER ESTE
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'], # VER ESTE (pode ficar um float dado por mim)
        'shrinking': [True, False],
        'probability': [True, False],
        'tol': [0.00001, 0.0001, 0.001], # VER ESTE
        'class_weight': ['dict', 'balanced', None],
        'max_iter': [-1],
        'decision_function_shape': ['ovo', 'ovr'],
        'break_ties': [True, False],
        'random_state': [0],
        'verbose': [0],
    }

    """
    return {
        'C': hp.quniform('C', 0.01, 3, 0.01),
        'kernel': hp.choice('kernel', ['rbf']),
        'gamma': hp.uniform('gamma',0.01, 0.8),
        'random_state': 0 
        }
    """

# Random Forest Tunning Parameters
def RandomForestParameters():

    return {
        'n_estimators': [50],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_weight_fraction_leaffloat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_features': ['sqrt', 'log2', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_impurity_decrease': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'bootstrap': [True, False],
        'oob_score': [True, False],
        'n_jobs': [-1],
        'random_state': [0],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'ccp_alpha': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_saples': [0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9],
        'verbose': [0],
    }

    """
    return {
        'bootstrap': hp.choice('bootstrap', [True]),
        'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(5, 9, 1, dtype=int)),
        'max_depth': hp.choice('max_depth',  np.arange(1, 9, 1, dtype=int)),
        'max_features': hp.choice('max_features',['log2',None,'sqrt']),
        'n_estimators': hp.choice('n_estimators', np.arange(10, 45, 3, dtype=int)),
        'criterion': hp.choice('criterion', ["entropy"]),
        'max_samples': hp.choice('max_samples', np.arange(0.8, 0.9, 0.01)),
        'n_jobs': -1,
        'random_state':0
    }
    """

# XGBoost Tunning Parameters    
def XGBParameters():

    return {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'validate_parameters': [True, False],
        'disable_default_eval_metric': [True, False],
        'eta': [0.01, 0.05, 0.1, 0.2, 0.3, 0,4], # VER ESTE
        'gamma': [0, 1, 5, 10, 20, 50, 100, 1000],
        'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'min_child_weight': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        'max_delta_step': [0, 1, 5, 10, 20, 50, 100, 1000],
        'subsample': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'sampling_method': ['uniform', 'gradient_based'],
        'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
        'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'lambda': [0.01, 0.1, 1, 5,8, 10, 13, 20],
        'alpha': [0.01, 0.1, 1, 5,8, 10, 13, 20],
        'tree_method': ['auto', 'exact', 'approx', 'hist', 'gpu_hist'],
        'process_type': ['default', 'update'],
        'grow_policy': ['depthwise', 'lossguide'],
        'verbose': [0],
    }

    """
    return {
        'learning_rate':    hp.uniform('learning_rate', 0.02, 0.4),
        'max_depth':        scope.int(hp.quniform('max_depth',3, 25, 2)),
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 4, 20, 2)),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.9, 0.02),
        'subsample':       hp.quniform('subsample', 0.6, 1, 0.03),
        'n_estimators':     scope.int(hp.quniform('n_estimators', 5, 80, 5)),
        'reg_alpha' : hp.choice('reg_alpha', [0.01, 0.1, 1, 5,8, 10, 13, 20]),
        'reg_lambda' : hp.choice('reg_lambda',  [0.01, 0.1, 1, 5,8, 10, 13, 20]),
        'random_state': 0,
        'n_jobs':-1,
        'use_label_encoder':False
    }
    """

# Selects the classifier to execute the tunning process
def SelectClassifier():

    if classifier == 'log_regr':
        return LogRegParameters()
    elif classifier == 'svm':
        return SvmParameters()
    elif classifier == 'rand_for':
        return RandomForestParameters()
    elif classifier == 'rand_for':
        return XGBParameters()
    else:
        exit("Classifier = {'log_regr', 'svm', 'rand_for', 'xgb'")

# Computes the grid search
def BestParameters_GridSearch():

    parameters = SelectClassifier()
    model = models[classifier]()

    print("\n Training Grid Search...")

    grid_search = GridSearchCV(model, param_grid = parameters, scoring = 'accuracy', n_jobs = -1, cv = 10)   

    grid_search.fit(xx, yy)

    print(grid_search.best_score_)
    print(grid_search.best_params_)


# ============================================================================================= #
# ========================================= HYPEROPT ========================================== #
# ============================================================================================= #

# Defines the parameters' set to optimize
def SearchSpace():
    if classifier == 'log_regr':
        return {
            "warm_start": hp.choice('warm_start', [True]),
            "tol": hp.quniform('tol', 0.00001, 0.001, 0.0005),
            "fit_intercept": hp.choice('fit_intercept', [False,True]),
            "C": hp.uniform('C', 0.01, 3),
            "solver": hp.choice('solver', ['lbfgs']),
            "penalty": hp.choice('penalty', ['l2']),
            "multi_class": 'ovr',
            "random_state": 0,
            "max_iter": 1000
            }
    elif classifier == 'svm':
        return {
            'C': hp.quniform('C', 0.01, 3, 0.01),
            'kernel': hp.choice('kernel', ['rbf']),
            'gamma': hp.uniform('gamma',0.01, 0.8),
            'random_state': 0 
        }
    elif classifier == 'rand_for':
        return {   
            'bootstrap': hp.choice('bootstrap', [True]),
            'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(5, 9, 1, dtype=int)),
            'max_depth': hp.choice('max_depth',  np.arange(1, 9, 1, dtype=int)),
            'max_features': hp.choice('max_features',['log2',None,'sqrt']),
            'n_estimators': hp.choice('n_estimators', np.arange(10, 45, 3, dtype=int)),
            'criterion': hp.choice('criterion', ["entropy"]),
            'max_samples': hp.choice('max_samples', np.arange(0.8, 0.9, 0.01)),
            'n_jobs': -1,
            'random_state':0
        }
    elif classifier == 'xgb':
        return {
            'learning_rate':    hp.uniform('learning_rate', 0.02, 0.4),
            'max_depth':        scope.int(hp.quniform('max_depth',3, 25, 2)),
            'min_child_weight': scope.int(hp.quniform('min_child_weight', 4, 20, 2)),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.9, 0.02),
            'subsample':       hp.quniform('subsample', 0.6, 1, 0.03),
            'n_estimators':     scope.int(hp.quniform('n_estimators', 5, 80, 5)),
            'reg_alpha' : hp.choice('reg_alpha', [0.01, 0.1, 1, 5,8, 10, 13, 20]),
            'reg_lambda' : hp.choice('reg_lambda',  [0.01, 0.1, 1, 5,8, 10, 13, 20]),
            'random_state': 0,
            'n_jobs':-1,
        }
    else:
        exit("Classifier = {'log_regr', 'svm', 'rand_for', 'xgb'")

# Define the hyperoptimization strategy   
def ModelDefinition(params):
    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    clf = models[classifier](**params)
    accuracy = cross_val_score(clf, xx, yy, cv = skf, scoring = 'accuracy')
    acc =  accuracy.mean()
    return {'loss': -acc, 'status': STATUS_OK}

# Define the hyperoptimization model   
def BestParameters_HyperOpt():

    best_params = fmin(
                fn = ModelDefinition,
                space = SearchSpace(),
                algo = tpe.suggest,
                max_evals = 150,
                trials = Trials(),
                rstate = np.random.default_rng(0),
                return_argmin = False,
                )

    print(best_params)


# ============================================================================================= #
# =========================================== MAIN ============================================ #
# ============================================================================================= #

# Number of values to ignore due to NaN values
ignore_nan = 300

# Define all models
models = {'log_regr' : LogRegr, 'svm' : SVC, 'rand_for' : RandomForestClassifier, 'xgb': XGBClassifier}

# Load yy_train
yy = LoadEnvironment('Renko10-20year-200-0', 'yy_train')


#############################################################
#################### Logistic Regression ####################
#############################################################

print('#################### Logistic Regression ####################')

# Load xx_train
xx = LoadEnvironment('20y-200-9-new-1', 'xx_train')
classifier = 'log_regr'

# Computes Hyper Optimization
print('Performing Hyper Optimization...')
BestParameters_HyperOpt()

# Computes Grid Search
#print('Performing Grid Search...')
#BestParameters_GridSearch()

#############################################################
####################### Random Forest #######################
#############################################################

print('####################### Random Forest #######################')

# Load xx_train
xx = LoadEnvironment('20y-200-7-new-1', 'xx_train')
classifier = 'rand_for'

# Computes Hyper Optimization
print('Performing Hyper Optimization...')
BestParameters_HyperOpt()

# Computes Grid Search
#print('Performing Grid Search...')
#BestParameters_GridSearch()

#############################################################
############################ XGB ############################
#############################################################

print('############################ XGB ############################')
# Load xx_train
xx = LoadEnvironment('20y-200-8-new-1', 'xx_train')
classifier = 'xgb'

# Computes Hyper Optimization
print('Performing Hyper Optimization...')
BestParameters_HyperOpt()

# Computes Grid Search
#print('Performing Grid Search...')
#BestParameters_GridSearch()

#############################################################
############################ SVM ############################
#############################################################

print('############################ SVM ############################')
# Load xx_train
xx = LoadEnvironment('20y-200-10-new-1', 'xx_train')
classifier = 'svm'

# Computes Hyper Optimization
print('Performing Hyper Optimization...')
BestParameters_HyperOpt()

# Computes Grid Search
#print('Performing Grid Search...')
#BestParameters_GridSearch()
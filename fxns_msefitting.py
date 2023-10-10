import math
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Lasso, LinearRegression, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from sklearn.decomposition import PCA

import fxns_preprocessing as pp 


# Verbose parameter for RF and GBT
set_verbose = 0


########################################################################################################
# MAIN FUNCTIONS
########################################################################################################

def calc_pooledMSE(data_in, method, folds = 5, parallel = False):
    # This function fits the model of choice on a pooled sample and calculates MSE
    #
    # data_in: pandas dataframe with data containing the sorting (iterating)
    #   variable (year) as the first column, and the dependent variable
    #   as the second column. All other variables are used as X's.
    # method: model choice - see options below
    # folds: number of CV folds if using CV method
    # parallel: uses all cores for CV when possible if True

    df = data_in.copy()
    Yvar = df.columns[0]
    # Preprocess data
    no_winsor = [x for x in df.columns if x == Yvar or 'TIME_' in x or 'SIC_' in x]
    processcols = [x for x in df.columns if x not in no_winsor]
    df = pp.clip_outlier_iqr(df, processcols, 5)
    df[processcols] = (df[processcols] - df[processcols].mean())/df[processcols].std()
    # Estimate specified model (and keep non-zero x variables for LASSO only)
    print('Fitting', method, 'using pooled sample...')
    if method == 'CV Lasso':
        [model, xvars] = run_lassoCV(df, postlasso = False, numCVs = folds, parallel = parallel)
    elif method == 'Iterative Lasso':
        [model, xvars] = run_iterativelasso(df, postlasso = False)
    elif method == 'CV Post-Lasso':
        [model, xvars] = run_lassoCV(df, postlasso = True, numCVs = folds, parallel = parallel)
    elif method == 'Iterative Post-Lasso':
        [model, xvars] = run_iterativelasso(df, postlasso = True)
    elif method == 'CV Ridge':
        model = run_ridgeCV(df, folds)
        xvars = False
    elif method == 'GCV Ridge':
        model = run_ridgeCV(df)
        xvars = False
    elif method == 'Elastic Net':
        model = run_elasticnetCV(df, numCVs = folds, parallel = parallel)
        xvars = False
    elif method == 'Random Forest':
        model = run_randomforestCV(df, numCVs = folds, parallel = parallel)
        xvars = False
    elif method == 'Gradient Boosted Tree':
        model = run_gradboostedtreeCV(df, numCVs = folds, parallel = parallel)
        xvars = False
    elif method == 'OLS' or method == 'FM':
        model = run_ols(df)
        xvars = False
    else:
        print('Need to specify one of available methods.')
        return
    print('Model fitted!')
    # Calculate MSE and store result
    [mse, forecasts] = calc_testMSE(df, model, xvars)
    return [mse, forecasts]

def calc_rollingMSE(data_in, method, roll_yrs = 0, pca = False, folds = 5, parallel = False):
    # This function fits the model of choice on a rolling basis, then calculates
    # the MSE.
    #
    # data_in: pandas dataframe with data containing the sorting (iterating)
    #   variable (year) as the first column, and the dependent variable
    #   as the second column. All other variables are used as X's.
    # method: model choice - see options below
    # pca:= True if you want to PCA X variables
    # folds: number of CV folds if using CV method
    # parallel: uses all cores for CV when possible if True

    df = data_in.copy()
    yrvar = df.columns[0]
    Yvar = df.columns[1]
    # Define columns for pre-processing
    no_winsor = [x for x in df.columns if x == Yvar or x == 'T_DATADATE' or 'SIC_' in x or 'YEAR_DATADATE' in x]
    processcols = [x for x in df.columns if x not in no_winsor]
    # Calculate first and last years
    startyear = np.min(df[yrvar]) + float(max(roll_yrs, 1)) # need at least one year
    endyear = max(df[yrvar])
    # Create result storage
    MSE = []
    forecasts_dfs = []
    # Create list of years to loop over
    years = list(range(int(startyear), int(endyear+1)))
    # Loop to fit models
    for yr in years:
        # Calculate start year
        if roll_yrs == 0:
            begyr = min(df[yrvar])
        else:
            begyr = yr - roll_yrs
        if method == 'FM': # this is the difference between OLS and FM - FM only uses most recent C/S
            begyr = yr - 1 
        # Get training and test datasets, dropping date column
        traindata0 = df[(df[yrvar] < yr) & (df[yrvar] >= begyr)].copy().drop(yrvar, axis = 1)
        testdata0 = df[(df[yrvar] == yr)].copy().drop(yrvar, axis = 1)
        # Preprocess data
        [traindata1, testdata1] = pp.joint_winsorize(traindata0, testdata0, processcols, 10)
        [traindata, testdata] = pp.joint_standardize(traindata1, testdata1, processcols)
        traindata[processcols] = traindata[processcols].fillna(0) # fill with zeros for any columns that have std=0 after winsorizing => nan
        testdata[processcols] = testdata[processcols].fillna(0)
        # Apply PCA, ensuring same PCA on training and testing
        if pca:
            pca = PCA(n_components = 0.95)
            non_pca = ['LEAD_EPS', 'F_ANALYST', 'PRCC', 'EPS', 'LAG_EPS', 'LAG2_EPS']
            pca_cols = [x for x in traindata.columns if x not in non_pca]
            traindata_np = traindata[non_pca].copy()
            testdata_np = testdata[non_pca].copy()
            traindata_p = pca.fit_transform(traindata[pca_cols])
            testdata_p = pca.transform(testdata[pca_cols])
            traindata = pd.concat([traindata_np, pd.DataFrame(traindata_p, index = traindata_np.index)], axis = 1)
            testdata = pd.concat([testdata_np, pd.DataFrame(testdata_p, index = testdata_np.index)], axis = 1)
        # Estimate specified model (and keep non-zero x variables for LASSO only)
        print('Fitting', method, 'using data prior to {}...'.format(yr))
        if method == 'CV Lasso':
            [model, xvars] = run_lassoCV(traindata, postlasso = False, numCVs = folds, parallel = parallel)
        elif method == 'Iterative Lasso':
            [model, xvars] = run_iterativelasso(traindata, postlasso = False)
        elif method == 'CV Post-Lasso':
            [model, xvars] = run_lassoCV(traindata, postlasso = True, numCVs = folds, parallel = parallel)
        elif method == 'Iterative Post-Lasso':
            [model, xvars] = run_iterativelasso(traindata, postlasso = True)
        elif method == 'CV Ridge':
            model = run_ridgeCV(traindata, folds)
            xvars = False
        elif method == 'GCV Ridge':
            model = run_ridgeCV(traindata)
            xvars = False
        elif method == 'Elastic Net':
            model = run_elasticnetCV(traindata, numCVs = folds, parallel = parallel)
            xvars = False
        elif method == 'Random Forest':
            model = run_randomforestCV(traindata, numCVs = folds, parallel = parallel)
            xvars = False
        elif method == 'Gradient Boosted Tree':
            model = run_gradboostedtreeCV(traindata, numCVs = folds, parallel = parallel)
            xvars = False
        elif method == 'OLS' or method == 'FM':
            model = run_ols(traindata)
            xvars = False
        else:
            print('Need to specify one of available methods.')
            return
        print(yr, 'model fitted!')
        # Calculate MSE and store result
        [mse, forecasts] = calc_testMSE(testdata, model, xvars)
        MSE.append(mse)
        # Format forecasts_dfs and store results
        forecasts['YEAR_DATADATE'] = yr
        forecasts_dfs.append(forecasts)
    # Format output of MSE
    output = [MSE, years]
    method_out = method.replace(' ', '').lower()
    label = 'MSE_'+method_out
    output_df = pd.DataFrame({'YEAR':output[1], label:output[0]})
    output_df.set_index('YEAR', drop = True, inplace = True)
    # Format output of forecasts
    forecasts_df_out = pd.concat(forecasts_dfs)
    return [output_df, forecasts_df_out]


# Function to calculate MSE and output dataframe of forecasts from any model
def calc_testMSE(testdata, model, nonzero_coefs = False):
    # This function calculates the MSE from a model fit in the loop part of calc_rollingMSE()
    # 
    # testdata: data with same format as traindata used to fit model above,
    # having the dependent variable as the first column
    # model: model fit on training data
    # nonzero_coefs: list of columns in testdata that are used as X's in 
    # the model.
    
    # Define variables
    if nonzero_coefs == False:
        X = np.array(testdata.drop(testdata.columns[0], axis = 1))
    else:
        X = np.array(testdata.drop(testdata.columns[0], axis = 1))[:, nonzero_coefs]
    Y = np.array(testdata.iloc[:,0]).reshape(-1,1)
    n = np.shape(X)[0]
    # Calculate MSE
    Yhat = model.predict(X).reshape(-1,1)
    MSE = 1/n*np.sum(np.power(Y - Yhat,2))
    # Dataframe of forecasts
    target = testdata.columns[0]
    output_df = testdata.copy()
    predict_label = 'PREDICTED_' + target
    output_df[predict_label] = Yhat
    output_df = output_df[[target, predict_label]].copy()
    return [MSE, output_df]



########################################################################################################
# RIDGE FUNCTIONS
########################################################################################################

def run_ridgeCV(traindata, cv = 0):
    # This function returns the estimated ridgemodel
    # where the penalty-level is chosen based CV, making. Everything done
    # here includes constants in estimations, so data does not need to
    # be demeaned.
    # CV = 0 corresponds to generalized cross validation
    #
    # Define variables
    X = np.array(traindata.drop(traindata.columns[0], axis = 1))
    Y = np.ravel(np.array(traindata.iloc[:,0]).reshape(-1,1))
    # Estimate ridge with CV or GCV
    alphas = [1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01,1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]
    if cv > 0:
        ridge = RidgeCV(alphas = alphas, cv = cv).fit(X,Y)
    else:
        # When no cv is specified, RidgeCV() defaults to generalized CV
        ridge = RidgeCV(alphas = alphas).fit(X,Y)
    return ridge



########################################################################################################
# LASSO FUNCTIONS
########################################################################################################

def get_lassolambda(traindata, conditional = True, c = 1.1, gamma = 0.05, phi = 0.1, \
                    nu = 1e-8, K = 20, sims = 1000, postlasso = False):
    # This function calculates the penalty level (multiplied by n), following
    # the iterative method used in Belloni and Chernozhukov 2013 - 
    # "HD Sparse Econometric Models". Many default values of variables come from
    # their reccomendations.
    #
    # traindata: pandas dataframe for training the model. This dataset must
    #   be standardized and have the dependent variable as the first column.
    # conditional: = True if you want to calculate the X-dependent penalty,
    #   otherwise normal  distribution is used
    # sims: number of simulations to estimate X-dependent penalty level
    # postlasso: indicator for if you want to use iterative post-Lasso or
    #   normal lasso
    
    # Define variables
    X = np.array(traindata.drop(traindata.columns[0], axis = 1))
    Y = np.array(traindata.iloc[:,0]).reshape(-1,1)
    [n,p] = np.shape(X)
    
    # Choose regressors for initalization - just pick first half if # regressors
    #   greater than half number of observations, including constant
    Xs = traindata.drop(traindata.columns[0], axis = 1).columns.tolist()
    if round(n/2) > p:
        I0 = Xs
    else:
        I0 = Xs[0:round(n/2)] 
    X0 = np.array(traindata[I0])
    
    # Initialization of sigma based on OLS residual std error
    sigma_past = math.sqrt(1/n*np.sum(np.power(Y - np.array(\
                            LinearRegression().fit(X0,Y).predict(X0)),2)))
    sigma_curr = phi*sigma_past
    
    # Get penalty level
    if conditional == True:
        # Calculate penalty level conditonal on X based on simulation
        Gamma = []
        for j in range(sims):
            g = np.random.normal(size = n)
            Gamma.append(n*(np.max(1/n* X.T @ g)))
        Gammaq = np.percentile(Gamma, 1-gamma)
    else:
        # Calculate penalty level independent of X using normal CDF
        Gammaq = norm.ppf(1-gamma/2/p)
    
    # While Loop running Lassos until convergence, or hit limit
    i = 1
    if postlasso == True:
        while abs(sigma_curr - sigma_past) > nu and i < K:
            # Estimate Lasso
            lambdai = 2*c*sigma_curr*Gammaq
            lasso = Lasso(alpha = lambdai/n).fit(X, Y)
            # Extract non-zero coefficents and run post-lasso OLS
            nonzero_coefs = list(np.nonzero(lasso.coef_)[0])
            shat = len(nonzero_coefs)
            X_nonzero = X[:,nonzero_coefs]
            ols = LinearRegression().fit(X_nonzero, Y)
            # Calculate SSE from post-lasso to get new sigma
            Yhat = ols.predict(X_nonzero).reshape(-1,1)
            sigma_past = sigma_curr
            sigma_curr = math.sqrt(1/n*np.sum(np.power(Y - Yhat,2)))*n/(n-shat)
            i += 1
    else:
        while abs(sigma_curr - sigma_past) > nu and i < K:
            # Estimate lasso using past sigma
            lambdai = 2*c*sigma_curr*Gammaq
            lasso = Lasso(alpha = lambdai/n).fit(X, Y)
            # Calculate SSR from lasso to get new sigma
            Yhat = lasso.predict(X).reshape(-1,1)
            sigma_past = sigma_curr
            sigma_curr = math.sqrt(1/n*np.sum(np.power(Y - Yhat,2)))
            i += 1
    
    # Return lambda after completing while loop
    lambdaoutput = 2*c*sigma_curr*Gammaq
    return [lambdaoutput, i-1]
    

def run_iterativelasso(traindata, postlasso = False):
    # This function returns the estimated lasso (or post lasso) model
    # where the penalty-level is chosen based on iterating, making
    # this an iterative lasso (or iterative-post lasso). Everything done
    # here includes constants in estimations, so data does not need to
    # be demeaned. The function also returns which columns of the 
    # data are used in the estimation, which is needed for prediction
    # for post-lasso where not all variables are used.
    #
    # Define variables
    X = np.array(traindata.drop(traindata.columns[0], axis = 1))
    Y = np.array(traindata.iloc[:,0]).reshape(-1,1)
    n = np.shape(X)[0]
    # Get penalty level from iteration
    [lambdap, numiters] = get_lassolambda(traindata, postlasso = postlasso)
    if postlasso == True:
        print('Iterative Post-Lasso took', numiters, 'iterations to find penalty level...')
    else:
        print('Iterative Lasso took', numiters, 'iterations to find penalty level...')
    # Estimate lasso
    lasso = Lasso(alpha = lambdap/n).fit(X, Y)
    print('Lasso has', np.shape(np.nonzero(lasso.coef_)[0])[0], 'out of', np.shape(X)[1], 'non-zero coefficients.')
    # Do post-lasso OLS regression and return model
    if postlasso == True:
        nonzero_coefs = list(np.nonzero(lasso.coef_)[0])
        X_nonzero = X[:,nonzero_coefs]
        ols = LinearRegression().fit(X_nonzero, Y)
        return [ols, nonzero_coefs]
    else:
        nonzero_coefs = [x for x in range(np.shape(X)[1])]
        return [lasso, nonzero_coefs]
        

def run_lassoCV(traindata, numCVs, postlasso = False, parallel = False):
    # This function returns the estimated lasso (or post lasso) model
    # where the penalty-level is chosen based CV, making. Everything done
    # here includes constants in estimations, so data does not need to
    # be demeaned. The function also returns which columns of the 
    # data are used in the estimation, which is needed for prediction
    # for post-lasso where not all variables are used.
    #
    # Define variables
    X = np.array(traindata.drop(traindata.columns[0], axis = 1))
    Y = np.ravel(np.array(traindata.iloc[:,0]).reshape(-1,1)) # need ravel for LASSO CV
    # Estimate LASSO with CV
    if parallel == True:
        lasso = LassoCV(cv = numCVs, random_state = 0, n_jobs=-1).fit(X,Y)
    else:
        lasso = LassoCV(cv = numCVs, random_state = 0).fit(X,Y)
    print('Lasso has', np.shape(np.nonzero(lasso.coef_)[0])[0], 'out of', np.shape(X)[1], 'non-zero coefficients.')
    # Do post-lasso OLS regression and return model
    if postlasso == True:
        nonzero_coefs = list(np.nonzero(lasso.coef_)[0])
        X_nonzero = X[:,nonzero_coefs]
        ols = LinearRegression().fit(X_nonzero, Y)
        return [ols, nonzero_coefs]
    else:
        nonzero_coefs = [x for x in range(np.shape(X)[1])]
        return [lasso, nonzero_coefs]



########################################################################################################
# ELASTIC NET FUNCTIONS
########################################################################################################

def run_elasticnetCV(traindata, numCVs, parallel = False):
    # This function returns the estimated elastic net model
    # where the penalty-level is chosen based CV. Everything done
    # here includes constants in estimations, so data does not need to
    # be demeaned.
    #
    # Define variables
    X = np.array(traindata.drop(traindata.columns[0], axis = 1))
    Y = np.ravel(np.array(traindata.iloc[:,0]).reshape(-1,1))
    # Estimate elastic net over grid of L1-L2 parameter ratios
    l1_ratio = [0.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 0.99]
    if parallel == True:
        en = ElasticNetCV(l1_ratio = l1_ratio, cv = numCVs, n_jobs=-1, random_state = 0).fit(X,Y) 
    else:
        en = ElasticNetCV(l1_ratio = l1_ratio, cv = numCVs, random_state=0).fit(X,Y) 
    print('Chosen L1 ratio from CV:', en.l1_ratio_)
    return en



########################################################################################################
# RANDOM FOREST FUNCTIONS
########################################################################################################

def run_randomforestCV(traindata, numCVs, parallel = False):
    # Define variables
    X = np.array(traindata.drop(traindata.columns[0], axis = 1))
    Y = np.ravel(np.array(traindata.iloc[:,0]).reshape(-1,1))
    # Cross-validate
    rf = RandomForestRegressor(random_state=0, bootstrap=True)
    grid = {
        'n_estimators': [1000],
        'max_depth': [4, 8, 12, 16],
        'max_features':[0.3, 0.6, 0.9, 1],
        'min_samples_leaf':[1, 3, 5],
        'min_samples_split':[2, 6, 10]
    }
    if parallel == True:
        grid_search = GridSearchCV(estimator = rf, param_grid = grid, cv = numCVs, n_jobs = -1, verbose = set_verbose)
    else:
        grid_search = GridSearchCV(estimator = rf, param_grid = grid, cv = numCVs, verbose = set_verbose)
    model = grid_search.fit(X,Y)
    print('Chosen parameters from CV:', model.best_params_)
    return model



########################################################################################################
# GRADIENT BOOSTED TREE FUNCTIONS FUNCTIONS
########################################################################################################

def run_gradboostedtreeCV(traindata, numCVs, parallel = False):
    # Define variables
    X = np.array(traindata.drop(traindata.columns[0], axis = 1))
    Y = np.ravel(np.array(traindata.iloc[:,0]).reshape(-1,1))
    # Cross-validate
    gbt = GradientBoostingRegressor(random_state=0)
    grid = {
        'n_estimators': [500, 1000, 5000, 10000],
        'max_depth': [1,2,3],
        'learning_rate': [0.001, 0.01, 0.1]
    }
    if parallel == True:
        grid_search = GridSearchCV(estimator = gbt, param_grid = grid, cv = numCVs, n_jobs = -1, verbose = set_verbose)
    else:
        grid_search = GridSearchCV(estimator = gbt, param_grid = grid, cv = numCVs, verbose = set_verbose)
    model = grid_search.fit(X,Y)
    print('Chosen parameters from CV:', model.best_params_)
    return model



########################################################################################################
# OLS FUNCTION
########################################################################################################

def run_ols(traindata):
    # This function runs OLS on the given pandas dataframe. The pandas dataframe
    # must have Y as the first column, and all other columns will be used as X's
    X = np.array(traindata.drop(traindata.columns[0], axis = 1))
    Y = np.array(traindata.iloc[:,0]).reshape(-1,1)
    ols = LinearRegression().fit(X, Y)
    return ols

# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize, Bounds, brute, fmin
import math
from scipy.stats import chi2
from sklearn.utils import resample
from tqdm import tqdm


#### SAMPLE MOMENT FUNCTION
# This function takes in the data, a (vector-valued) score function, and
# the gradient of that score function. It then returns the sample moment
# vector g_T, and the sample moment jacobian G_T. 
def gT_gmm(X, theta, h, dh_dtheta):
    # X: data, where rows are observations and columns are variables, must
    #    be a numpy array. make sure data is ordered correctly according
    #    h function
    # theta: parameters
    # h: score function
    # dh_dtheta: derivative of score function (transposed)
    T = X.shape[0]
    scores = h(X, theta)
    gT = np.sum(scores, axis = 0)/T
    if dh_dtheta != None:
        scorederivs = dh_dtheta(X, theta)
        dT = np.sum(scorederivs, axis = 0)/T
    else:
        dT = ''
    return [gT, dT]

#### GMM OBJECTIVE FUNCTION
    # This function takes in the same variables as gT_gmm, in addition to a
    # weigthing matrix W. It returns the GMM objective function, and it's 
    # derivative. 
    # bootadj: adjustment for when you're doing a quick or slow bootstrap
def gmmobjfxn(theta, h, X, W, dh_dtheta, bootadj = 0):
    inputs =  gT_gmm(X, theta, h, dh_dtheta)
    g_T = inputs[0]
    d_T = inputs[1]
    objfxn = np.dot(0.5, np.transpose(g_T) @ W @ g_T)
    if dh_dtheta != None:
        dobjfxn_dtheta = d_T @ W @ g_T
        return [objfxn - bootadj, dobjfxn_dtheta]
    else:
        return objfxn - bootadj
    
#### NEWEY-WEST SPECTRAL DENSITY ESTIMATOR
def neweywest(z, S_T):
    # z: GMM residual matrix with rows as observations and columns as
    #    different residuals. Must be mean zero and a numpy array.
    # S_T: truncation parameter for Newey-West
    T = np.shape(z)[0]
    cols = np.shape(z)[1]
    sumindex = np.linspace(-S_T, S_T, (2*S_T+1)).astype(int)
    sumterms = np.zeros((len(sumindex), cols, cols))
    for i in sumindex:
        zt_zti = np.zeros((T-abs(i), cols, cols))
        for t in range(abs(i),T):
            zt = z[t,:]
            zti = z[(t-abs(i)),:]
            zt_zti[(t-abs(i)),:,:] = np.outer(zt, zti)
        sum_ztzti = np.sum(zt_zti, axis = 0)
        sumterm_i = (1-abs(i/S_T))*(1/(T-i))*sum_ztzti
        sumterms[(i+S_T),:,:] = sumterm_i
    Omegahat = np.sum(sumterms, axis = 0)
    return Omegahat

#### IID RESIDUAL ESTIMATOR FUNCTION
def variance_iid(z):
    # z: GMM residual matrix with rows as observations and columns as
    #    different residuals. Must be mean zero and a numpy array.
    T = np.shape(z)[0]
    cols = np.shape(z)[1]
    sumterms = np.zeros((T, cols, cols))
    for t in range(T):
        zt = z[t,:]
        sumterms[t] = np.outer(zt,zt)
    Omegahat = np.sum(sumterms, axis = 0) / T
    return Omegahat

#### FUNCTION TO CALCULATE OMEGA AND OMEGAINV
def calc_Omega(score, data, thetahat, se, invert):
    T = len(data)
    ut = score(data, thetahat)
    ut_demean = ut - np.mean(ut, axis = 0)
    if se == 'neweywest':
        truncation = math.ceil(1.3*(T**(1/2))) # Lazarus et al (2019) truncation
        Omega = neweywest(ut_demean, truncation)
    else:
        Omega = variance_iid(ut_demean)
    if invert:
        Omegainv = np.linalg.pinv(Omega)
        return Omegainv
    else:
        return Omega

#### GMM ESTIMATION FUNCTION
def gmm(score, data, theta_init, W_init = None, se = 'iid', twostep = True, iterated = False, scorederiv = None, bootstrap = 'robust', 
        bounds_list = None, bootdata = None, N_boot = 0, constraints = None):
   #: W_init: Matrix for first step GMM
   #: theta_init: Initial guess for optimization = None if you want global optimization
   # data: data, where rows are observations and columns are variables, must
   #    be a numpy array. make sure data is ordered correctly according
   #    h function
   # score: score function (h in fxns above) - must be vectorized
   # scorederiv: derivative of score function (scorederiv in fxns above) = must be vectorized
   # twostep: = True if you want twostep, False otherwise
   # iterated: = True if you want iterated GMM
   # bootstrap: method for bootstrapping
   # bounds_list: bounds for optimization (will change method used from Nelder-Mead) as form [list lower bounds, list upper bounds]
   # bootdata: bootstrapped data as list of arrays (in case you did a cluster bootstrap before)
   # N_boot: number of bootstraps if bootdata not specified
   # constraints: any constraints you want in optimization

    # Determine whether Jacobian was given
    jac_in = scorederiv != None

   # Deal with bounds and choose optimization algorithm
    if bounds_list != None:
        if theta_init != None: # this will be indicator for knowing you're global
            bounds = Bounds(bounds_list[0], bounds_list[1])
            opmethod = 'trust-constr'
            constraints = None
            print('You are doing optimization with bounds.')
        else: # global optimization
            bounds = None
            slices = [slice(bounds_list[0][s], bounds_list[1][s], bounds_list[2][s]) for s in range(len(bounds_list[0]))]
            print('You are using brute force optimization for step one.')
            opmethod = 'Nelder-Mead' # Nelder-Mead for location optimization
            constraints = None
    else:
        if constraints == None:
            if jac_in:
                opmethod = 'BFGS'
            else:
                opmethod = 'Nelder-Mead' # Nelder-Mead for location optimization
            bounds = None
            print('You are doing optimization via ' + opmethod + ' without bounds.')
        else:
            opmethod = 'SLSQP'
            bounds = None
            print('You are doing optimization with constraints using ' + opmethod)

    # Set up initialization of W matrix to idenity, if one is not provided
    if W_init is None:
        if theta_init != None:
            W_init = np.identity(np.shape(score(data, theta_init))[1])
        else:
            print("If you're not going to provide a initialization of parameter vector, you need to provide an initialization of weighting matrix.")
    
    # Define minimization function that will be used throughout
    def gmm_minimize(df, theta0, W, offset=0):
        if constraints != None:
            out = minimize(fun = gmmobjfxn, x0 = theta0, args = (score, df, W, scorederiv, offset), jac = jac_in, method = opmethod, bounds = bounds, constraints = constraints)
        else:
            out = minimize(fun = gmmobjfxn, x0 = theta0, args = (score, df, W, scorederiv, offset), jac = jac_in, method = opmethod, bounds = bounds)
        return out
    
    # Iterated GMM function
    def iterate_gmm(theta_old, in_gmm, df, se, offset = 0, tol = .01, print_progress = False):
        if print_progress:
            print('Starting iterative GMM...')
        sol = in_gmm
        theta_new = in_gmm.x
        diff_thetas = np.max(np.divide(np.abs(np.subtract(theta_old, theta_new)), theta_old))
        i = 0
        while diff_thetas > tol:
            theta_old = theta_new
            Omegainv = calc_Omega(score, df, theta_old, se, invert = True)
            sol = gmm_minimize(data, theta_old, Omegainv, offset = offset)
            theta_new = sol.x
            diff_thetas = np.max(np.divide(np.abs(np.subtract(theta_new, theta_old)), theta_old))
            i += 1
        if print_progress:
            print('Iterated GMM finished after', i, 'iterations.')
        return sol

    # Step one
    if theta_init != None:
        print('Doing step one estimation locally...')
        stepone = gmm_minimize(data, theta_init, W_init)
        thetahat1 = stepone.x
    else:
        print('Doing step one estimation globally...')
        stepone = brute(func = gmmobjfxn, ranges = slices, args = (score, data, W_init, scorederiv),  full_output = True, finish = fmin)
        thetahat1 = stepone[0]
    print('Step one finished!')

    # Get problem dimensions
    T = len(data)
    dim_score = np.shape(score(data, thetahat1))[1]

    if twostep == False and iterated == False:
        # Calculate asymptotic variance for step one
        out_result = stepone
        thetahat2 = np.zeros(np.shape(thetahat1))
        if scorederiv != None:
            Omega = calc_Omega(score, data, thetahat1, se, invert = False)
            d_T = np.sum(scorederiv(data, thetahat1), axis = 0) / T
            V = np.linalg.pinv(d_T @ W_init @ np.transpose(d_T)) @ d_T @ W_init @ Omega @ W_init @ np.transpose(d_T) @ np.linalg.pinv(d_T @ W_init @ np.transpose(d_T))
            theta_bs = np.zeros(np.shape(thetahat1)[0])
        else:
            if N_boot > 0:
                if bootdata == None:
                    bootdata = [resample(data, replace = True) for i in range(N_boot)]
                else:
                    N_boot = len(bootdata)
                print('Doing', bootstrap, 'bootstrap with', N_boot, 'iterations...')
                if bootstrap == 'robust':
                    gmm_offset = 0 # robust bootstrap doesn't center => insensitive to misspec
                else:
                    gmm_offset = stepone.fun # slow bootstrap centers
                theta_bs = []
                for data_bs in tqdm(bootdata): # never use brute force for bootstrap
                    theta_bs.append(gmm_minimize(data_bs, thetahat1, W_init, gmm_offset).x)
                print('Bootstrap using', bootstrap, 'method complete!')
                theta_bs_c = [np.subtract(t, thetahat1) for t in theta_bs] # center around sample estimate
                theta_bs_outer = [np.outer(t,t) for t in theta_bs_c]
                V = T * 1/N_boot * sum(theta_bs_outer)
            else:
                V = np.zeros((np.shape(thetahat1)[0], np.shape(thetahat1)[0]))
                theta_bs = np.zeros(np.shape(thetahat1)[0])
        Jstat = 0
        pval = 0
    else:
        # Step two
        Omegainv = calc_Omega(score, data, thetahat1, se, invert = True)
        print('Doing step two estimation locally...')
        steptwo = gmm_minimize(data, thetahat1, Omegainv)
        print('Step two finished!')
        # Iterated GMM
        if iterated:
            steptwo = iterate_gmm(thetahat1, steptwo, data, se, print_progress = True)
        thetahat2 = steptwo.x
        f_step2 = steptwo.fun
        out_result = steptwo

        # Calculate asymptotic variance
        if scorederiv != None and N_boot == 0:
            d_T = np.sum(scorederiv(data, thetahat2), axis = 0) / T
            V = np.linalg.pinv(d_T @ Omegainv @ np.transpose(d_T))
            theta_bs = np.zeros(np.shape(thetahat1)[0])
        else:
            if N_boot > 0:
                if bootdata == None:
                    bootdata = [resample(data, replace = True) for i in range(N_boot)]
                else:
                    N_boot = len(bootdata)
                print('Doing', bootstrap, 'bootstrap with', N_boot, 'iterations...')
                if bootstrap == 'robust':
                    gmm_offset = 0 # robust bootstrap doesn't center => insensitive to misspec
                else:
                    bootstrap = 'slow'
                    gmm_offset = steptwo.fun # slow bootstrap centers
                theta_bs = []
                for data_bs in tqdm(bootdata): # never use brute force for bootstrap
                    thetahat1_bs = gmm_minimize(data_bs, thetahat2, W_init).x
                    Omegainv_bs = calc_Omega(score, data_bs, thetahat1_bs, se, invert = True)
                    gmm_bs = gmm_minimize(data_bs, thetahat1_bs, Omegainv_bs, gmm_offset)
                    # Iterated GMM for bootstrap if requested
                    if iterated:
                        gmm_bs = iterate_gmm(thetahat1_bs, gmm_bs, data_bs, se, offset = gmm_offset)
                    theta_bs.append(gmm_bs.x)
                print('Bootstrap using', bootstrap, 'method complete!')
                theta_bs_c = [np.subtract(t, thetahat2) for t in theta_bs] # center around sample estimate
                theta_bs_outer = [np.outer(t,t) for t in theta_bs_c]
                V = T * 1/N_boot * sum(theta_bs_outer)
            else:
                V = np.zeros((np.shape(thetahat2)[0], np.shape(thetahat2)[0]))
                theta_bs = np.zeros(np.shape(thetahat1)[0])
        Jstat = T * f_step2
        Jdof = dim_score - np.shape(theta_init)[0] # degree of over-ID
        pval = chi2.sf(Jstat, Jdof)

    # Calculate standard errors
    if np.shape(V)[0] == 1: # need this for 1-D case
        SEs = math.sqrt(V/T)
    else: 
        SEs = np.sqrt(np.diag(V)/T)
    
    # Output
    return [thetahat1, thetahat2, SEs, Jstat, pval, out_result, theta_bs]

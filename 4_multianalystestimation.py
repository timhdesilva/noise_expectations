##############################################################################
# TO BE SET BY USER

# Working directories
maindir = ''
forecastdir = maindir + ''
datadir = maindir + ''
outputdir = maindir + ''
codedir = maindir + ''
figdir = maindir + ''
##############################################################################

import pandas as pd
import numpy as np
import os
import random
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import importlib
from scipy.optimize import basinhopping
import pickle
from tabulate import tabulate
from functools import reduce
from sklearn.utils import resample

random.seed(33)

##############################################################################
## FILE PARAMETERS
##############################################################################

# Global bootstrap parameters
Nb = 200
boot_id = 'i'
global_bootstrap = boot_id != '' and Nb > 0


##############################################################################
## USEFUL FUNCTIONS
##############################################################################

# Import GMM Code
import fxns_gmm as fgmm
importlib.reload(fgmm)

# Function for nC2
def nC2(x):
    return math.comb(x,2)

# Function to output all possible two way products from a 1D array
def multiply_nC2(array, ids = None):
    N0 = np.shape(array)[0]
    N1 = nC2(N0)
    out = np.zeros(N1)
    id1s = np.zeros(N1)
    id2s = np.zeros(N1)
    counter = 0
    for id1 in range(N0):
        for id2 in range(id1 + 1, N0):
            out[counter] = array[id1]*array[id2]
            if ids is not None:
                id1s[counter] = ids[id1]
                id2s[counter] = ids[id2]
            counter += 1
    if ids is not None:
        return [out, id1s, id2s]
    else:
        return out

# Function for clipping outliers on iqr
def clip_outlier_iqr(df_in, cols, n_iqr = 5):
    df = df_in.copy()
    if isinstance(cols, str):
        cols = [cols]
    q1 = df[cols].quantile(0.25)
    q2 = df[cols].quantile(0.5)
    q3 = df[cols].quantile(0.75)
    mins = q2 - n_iqr * (q3-q1)
    maxs = q2 + n_iqr * (q3-q1)
    df[cols] = df[cols].clip(mins, maxs, axis=1)
    return df

# Function for winsorizing
def remove_outlier(df_in, cols, n_iqr = 5):
    df = df_in.copy()
    if isinstance(cols, str):
        cols = [cols]
    for col_name in cols:
        q1 = df[col_name].quantile(0.25)
        q2 = df[col_name].quantile(0.50)
        q3 = df[col_name].quantile(0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q2-n_iqr*iqr
        fence_high = q2+n_iqr*iqr
        df = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df

# Function to bootstrap single dataset
def bootstrap_single(data, clusterby = None, rs = 33):
    if clusterby == None:
        data_bs = resample(data, replace = True)
    else:
        ids = data[[clusterby]].drop_duplicates()
        data_bs = ids.sample(frac = 1, replace = True, random_state = rs)
        data_bs = data_bs.merge(data, on = clusterby, how = 'left')
    return data_bs.reset_index()

# Function to bootstrap
def bootstrap_data(data, N_boot = 1000, clusterby = None):
    print('Bootstrapping data...')
    out = [bootstrap_single(data, clusterby, i) for i in range(N_boot)]
    print('Finished bootstrapping!')
    return out

# Function to do weighted average
def pd_w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

# Function for running OLS
def run_OLS(data_in, xvars, yvar, print_reg = False, keep_mseresid = False, return_model = False, weightvar = None, no_intercept = False):
    data = data_in.copy()
    regvars = xvars.copy()
    regvars.append(yvar)
    if weightvar is not None: # transform variables so you equivalently run a weighed least squares by weightvar
        data[regvars] = data[regvars].multiply(np.sqrt(data[weightvar]), axis = 'index')
        regvars.append(weightvar)
    data_nona = data[regvars].dropna()
    X = data_nona[xvars].copy()
    Y = data_nona[yvar].copy()
    if no_intercept == False:
        if weightvar is not None:
            X['Intercept'] = np.sqrt(data_nona[weightvar])
        else:
            X['Intercept'] = 1
    model = sm.OLS(Y,X).fit()
    if print_reg:
        print(model.summary())
    coefs = model.params
    ses = model.HC0_se
    adjrsq = model.rsquared_adj
    n = round(model.nobs)
    output = pd.DataFrame({'coef':coefs, 'se': ses, 'adjRsq':adjrsq, 'nobs':n})
    if keep_mseresid:
        output['mse_resid'] = model.mse_resid
    if return_model:
        return [output, model]
    else:
        return output

# Function to bin data
def bin_data(data_in, xvar, nbins):
    data = data_in.copy()
    bin_lengths = []
    bin_lab = 'bin_' + xvar
    if type(nbins) == int:
        if nbins > 0: # if zero, then the bins are already made
            try:
                data[bin_lab] = pd.qcut(data[xvar], nbins, labels = range(1,nbins + 1))
                bin_lengths.append(nbins)
            except:
                print('Had to drop duplicates.')
                data[bin_lab] = pd.qcut(data[xvar], nbins, duplicates = 'drop')
                data[bin_lab] = data[bin_lab].cat.rename_categories(range(1,1+len(data[bin_lab].unique())))
                bin_lengths.append(len(data[bin_lab].unique()))
        else:
            bin_lengths.append(data[xvar].nunique())
            data[bin_lab] = data[xvar].copy()
    return data

# Function for outputing latex table of summary stats
def summary_stats_table(data, sumvars, outfile, table_dict, percs = [0.05, 0.25, 0.5, 0.75, 0.95], ndig = 3, long = True, include_t = True):
    sumstats = data[sumvars].describe(percentiles = percs)
    sumstats.drop(index = ['max', 'min'], inplace = True) # drop mins and maxes - specify 100% or 0% if you want them
    if include_t:
        sumstats.loc['t-stat'] = sumstats.loc['mean']/sumstats.loc['std']*(sumstats.loc['count'].apply(lambda x: np.sqrt(x)))
    sumstats = sumstats.round(ndig)
    sumstats.loc[' '] = sumstats.columns
    if include_t:
        sorts = [sumstats.index[-1]] + list(sumstats.index[0:3]) + [sumstats.index[-2]] + list(sumstats.index[3:-2])
    else:
        sorts = [sumstats.index[-1]] + list(sumstats.index[0:3]) + list(sumstats.index[3:-1])
    sumstats = sumstats.loc[sorts,:]
    if not long:
        sumstats = tabulate(sumstats, tablefmt = 'latex_booktabs', numalign = 'center', floatfmt = ",." + str(ndig) + "f")
    else:
        sumstats = tabulate(sumstats.transpose(), tablefmt = 'latex_booktabs', numalign = 'center', floatfmt = ",." + str(ndig) + "f", headers = "keys", showindex = False)
    sumstats = sumstats.replace("." + ndig * "0", "")
    if table_dict != None:
        table_dict2 = table_dict.copy()
        if not long:
            table_dict2['count  &'] = '\\midrule \n Count &'
        else:
            table_dict2['count'] = "Count"
        table_dict2['mean'] = 'Mean'
        table_dict2['std'] = 'SD'
        table_dict2['t-stat'] = '$t$-stat'
        table_dict2['100\%'] = 'Max'
        table_dict2[' 0\%'] = ' Min'
        for find, rep in table_dict2.items():
            sumstats = sumstats.replace(find, rep)
    print(sumstats)
    if outfile != '':
        text_file = open(outfile, "w")
        n = text_file.write(sumstats)
        text_file.close()


##############################################################################
## FUNCTIONS FOR DATA CONSTRUCTION
##############################################################################

# Function to output summary stats
def make_sumstats(variables, labels):
    # Get all dfs
    os.chdir(datadir)
    dfs_it = []
    dfs_itj = []
    dictlab = {}
    for freq in ['q', 'a']:
        for horizon in range(1,5):
            # Read in DFs
            data_fy_lab = 'datait_' + freq + str(horizon) + '.sas7bdat'
            data_fy = pd.read_sas(data_fy_lab, format = 'sas7bdat', encoding = 'latin-1')
            data_fy.columns = map(str.upper, data_fy.columns)
            data_fy['AT'] = np.exp(data_fy['LOG_AT'])/1000
            data_fy = data_fy[['GVKEY', 'PENDS', 'LEAD_EPS', 'N_ANALYSTS', 'PRCC'] + variables]
            data_fyj = pd.read_sas(data_fy_lab.replace('it', 'itj'), format = 'sas7bdat', encoding = 'latin-1')
            data_fyj.columns = map(str.upper, data_fyj.columns)
            data_fyj = data_fyj[['GVKEY', 'PENDS', 'ANALYS', 'FCAST']].merge(data_fy[['GVKEY', 'PENDS', 'LEAD_EPS', 'PRCC']])
            # Calculate variables of interest in both datasets
            addon = '_' + freq + str(horizon)
            data_fyj['ERROR' + addon] = (data_fyj['LEAD_EPS'] - data_fyj['FCAST'])/data_fyj['PRCC']
            data_fyj.drop(columns = ['LEAD_EPS', 'FCAST', 'PRCC'], inplace = True)
            # Rename columns in it dataset
            data_fy.drop(columns = ['LEAD_EPS', 'PRCC'], inplace = True)
            data_fy.columns = ['GVKEY', 'PENDS'] + [x + addon for x in data_fy.columns[2:]]
            # Append data
            dfs_it.append(data_fy)
            dfs_itj.append(data_fyj)
            # Append labels
            if freq == 'q':
                strh = str(horizon/4)
                if horizon == 4:
                    strh = "{1^*}"
            else:
                strh = str(horizon)
            power = '^{h=' + strh + '}'

            dictlab["N\_ANALYSTS\_" + freq + str(horizon)] = '$N_{it}' + power + '$'
            for i in range(len(variables)):
                dictlab[variables[i].replace('_', '\_') + '\_' + freq + str(horizon)] = '$' + labels[i] + power + '$'
            dictlab["ERROR\_" + freq + str(horizon)] = '$\pi_{it+h}' + power + ' - F_t^j\pi_{it+h}' + power + '$'
    # Combine datasets
    datait = reduce(lambda x, y: pd.merge(x, y, on = ['GVKEY', 'PENDS'], how = 'outer'), dfs_it)
    dataitj = reduce(lambda x, y: pd.merge(x, y, on = ['GVKEY', 'PENDS', 'ANALYS'], how = 'outer'), dfs_itj)
    # Output summary stats
    os.chdir(outputdir)
    summary_stats_table(datait, datait.columns[2:], 'summarystats_it.tex', dictlab, percs = [0.10, 0.25, 0.5, 0.75, 0.90], ndig = 3, long = True, include_t = False)
    summary_stats_table(dataitj, dataitj.columns[3:], 'summarystats_itj.tex', dictlab, percs = [0.10, 0.25, 0.5, 0.75, 0.90], ndig = 3, long = True, include_t = False)

# Function to make dataset for estimation
def make_df(estimator, freq, horizon, W, scaled_p=True, iqr_drop=5, drop_winsor=True):
    # Set files to read in
    if '5_int0' in estimator and freq == 'a' and horizon >= 3:
        infile_f = 'forecastdf_' + freq + str(horizon) + '_' + estimator.replace('5_int0', '0_int0') + '_sic2'
    else:
        infile_f = 'forecastdf_' + freq + str(horizon) + '_' + estimator + '_sic2'
    if scaled_p:
        infile_f += '_scaled'
    infile_f += '.dta'
    os.chdir(forecastdir)
    f_e = pd.read_stata(infile_f)
    f_e_f = pd.read_stata(infile_f.replace('forecastdf', 'forecastdf_af'))
    f_e.rename(columns = {'PREDICTED_LEAD_EPS':'Fe'}, inplace = True)
    f_e_f.rename(columns = {'PREDICTED_F_ANALYST':'Fe_f', 'F_ANALYST':'Fa'}, inplace = True)
    f_e_f = f_e_f[['GVKEY',  'LEAD_PENDS', 'Fe_f', 'Fa']]
    comb = f_e.merge(f_e_f, on = ['GVKEY', 'LEAD_PENDS'], how = 'inner')
    # Get clustering variables and prices
    os.chdir(datadir)
    data_fy_lab = 'datait_' + freq + str(horizon) + '.sas7bdat'
    data_fy = pd.read_sas(data_fy_lab, format = 'sas7bdat', encoding = 'latin-1')
    data_fy.columns = map(str.upper, data_fy.columns)
    merge_cols = [x for x in data_fy.columns if x not in comb.columns and x != 'ANALYS'] + ['GVKEY', 'LEAD_PENDS']
    comb = comb.merge(data_fy[merge_cols], on = ['GVKEY', 'LEAD_PENDS'])
    # Merge in analyst level forecasts
    os.chdir(datadir)
    data_fyj_lab = data_fy_lab.replace('datait', 'dataitj')
    data_fyj = pd.read_sas(data_fyj_lab, format = 'sas7bdat', encoding = 'latin-1')
    data_fyj.columns = map(str.upper, data_fyj.columns)
    data_fyj = data_fyj[['GVKEY', 'PENDS', 'ANALYS', 'FCAST']].rename(columns = {'FCAST':'F_ijt'})
    comb = data_fyj.merge(comb, on = ['GVKEY', 'PENDS'])
    # Create bins of variables of interest
    comb0_it = comb.drop_duplicates(subset = ['GVKEY', 'LEAD_PENDS'])
    if W[0] != '':
        Wbins = []
        for Wi in W:
            comb_it = bin_data(comb0_it, Wi, 5)
            comb_it = comb_it[['GVKEY', 'LEAD_PENDS']].join(pd.get_dummies(comb_it['bin_' + Wi], prefix = 'bin_' + Wi))
            Wbins = [x for x in comb_it.columns if 'bin_'+Wi in x]
            comb = comb.merge(comb_it[['GVKEY', 'LEAD_PENDS'] + Wbins])
        W = [x for x in comb.columns if 'bin' in x]
    # Keep only relevent variables and add constants
    keeps = ['GVKEY', 'T_DATADATE', 'ANALYS', 'F_ijt', 'Fa', 'Fe_f', 'LEAD_EPS', 'Fe', 'LEAD_PENDS', 'N_ANALYSTS', 'PRCC']
    if W[0] != '':
        keeps = keeps + W
    keeps = list(set(keeps)) # drop any duplicates
    comb = comb[keeps]
    comb = comb.rename(columns = {'GVKEY':'i', 'T_DATADATE':'t', 'ANALYS':'j', 'Fa':'F_it',
                                  'Fe_f':'Fe_f_it', 'LEAD_EPS':'E_it', 'Fe':'Fe_it'})
    # Form year variable
    if freq == 'a':
        comb['year'] = comb['t']
    elif freq == 'q':
        comb['year'] = comb['t']
    # Normalize by price
    cols = ['F_ijt']
    if not scaled_p:
        cols += ['F_ijt', 'F_it', 'Fe_f_it', 'E_it', 'Fe_it',]
    comb[cols] = comb[cols].div(comb['PRCC'], axis = 'index')
    comb.drop(columns = 'PRCC', inplace = True)
    # Calculate variables needed for moment formation
    comb['F*_ijt'] = comb['F_ijt'] - comb['Fe_f_it']
    comb['E*_it'] = comb['E_it'] - comb['Fe_it']
    comb['delta'] = (comb['Fe_f_it'] - comb['Fe_it'])**2
    # Output dataset to stata
    os.chdir(datadir)
    stata = comb.copy()
    stata.columns = [x.replace('*','star').replace('^2','_sq') for x in comb.columns]
    stata.to_stata('GMMdata_'+ freq + str(horizon) + '.dta')
    # Rename Ws and add constants
    comb['Y1'] = 1
    comb['W1'] = 1
    if W[0] != '':
        for j in range(len(W)):
            comb = comb.rename(columns = {W[j]:'W' + str(j+2)})
    # Drop outliers
    w_cols = ['F*_ijt', 'E*_it', 'delta']
    controls = [x for x in comb.columns if 'W' in x or 'Y' in x]
    controls.sort()
    print('Original sample size:', np.shape(comb)[0])
    if iqr_drop > 0:
        if drop_winsor == True:
            comb = remove_outlier(comb, w_cols, iqr_drop)
            print('Observations remaining after dropping outliers:', np.shape(comb)[0])
        else:
            comb = clip_outlier_iqr(comb, w_cols + controls, iqr_drop)
    # Create interaction variables and list variables to keep
    comb['F*_ijtE*_it'] = comb['F*_ijt']*comb['E*_it']
    comb['F*_ijt^2'] = comb['F*_ijt']**2
    fcasts = ['F*_ijt', 'delta', 'F*_ijt^2', 'F*_ijtE*_it', 'F_ijt', 'Fe_it']
    dfout = comb[['i', 'j', 't', 'year', 'LEAD_PENDS', 'E_it', 'E*_it'] + fcasts + controls].copy()
    comb['F*_it'] = comb.groupby(['i', 'LEAD_PENDS'])['F*_ijt'].transform('mean') # mean of individual forecasts
    comb['F*_it^2'] = comb['F*_it']**2
    comb_it = comb.groupby(['i', 'LEAD_PENDS'])[['delta', 'F*_it', 'E*_it', 'E_it', 'N_ANALYSTS', 'F*_it^2']].median()
    comb_it['Ninv'] = 1/comb_it['N_ANALYSTS']
    return dfout

# Function to get datasets from multiple horizons, matched based on matchtype
def make_dfs_matched(matchid, across_freq, estimator, W, include_4, scaled_p=True, iqr_drop=5, drop_winsor=True):
    if include_4:
        # Get all dfs
        [dfq1, dfq2, dfq3, dfq4] = [make_df(estimator, 'q', h, W, scaled_p, iqr_drop, drop_winsor) for h in range(1,5)]
        [dfa1, dfa2, dfa3, dfa4] = [make_df(estimator, 'a', h, W, scaled_p, iqr_drop, drop_winsor) for h in range(1,5)]
        # Match datasets
        if matchid != ['']:
            # Quartely
            dfhq1 = dfq1.merge(dfq2[matchid].drop_duplicates()).merge(dfq3[matchid].drop_duplicates()).merge(dfq4[matchid].drop_duplicates())
            dfhq2 = dfq2.merge(dfq1[matchid].drop_duplicates()).merge(dfq3[matchid].drop_duplicates()).merge(dfq4[matchid].drop_duplicates())
            dfhq3 = dfq3.merge(dfq1[matchid].drop_duplicates()).merge(dfq2[matchid].drop_duplicates()).merge(dfq4[matchid].drop_duplicates())
            dfhq4 = dfq4.merge(dfq1[matchid].drop_duplicates()).merge(dfq2[matchid].drop_duplicates()).merge(dfq3[matchid].drop_duplicates())
            # Annual
            dfha1 = dfa1.merge(dfa2[matchid].drop_duplicates()).merge(dfa3[matchid].drop_duplicates()).merge(dfa4[matchid].drop_duplicates())
            dfha2 = dfa2.merge(dfa1[matchid].drop_duplicates()).merge(dfa3[matchid].drop_duplicates()).merge(dfa4[matchid].drop_duplicates())
            dfha3 = dfa3.merge(dfa1[matchid].drop_duplicates()).merge(dfa2[matchid].drop_duplicates()).merge(dfa4[matchid].drop_duplicates())
            dfha4 = dfa4.merge(dfa1[matchid].drop_duplicates()).merge(dfa2[matchid].drop_duplicates()).merge(dfa3[matchid].drop_duplicates())
            if across_freq: # Merge annual and quarterly togethers
                acrossid = [x if x != 't' else 'year' for x in matchid]
                ids = dfha1[acrossid].drop_duplicates().merge(dfhq1[acrossid].drop_duplicates())
                dfhq1 = dfhq1.merge(ids)
                dfhq2 = dfhq2.merge(ids)
                dfhq3 = dfhq3.merge(ids)
                dfhq4 = dfhq4.merge(ids)
                dfha1 = dfha1.merge(ids)
                dfha2 = dfha2.merge(ids)
                dfha3 = dfha3.merge(ids)
                dfha4 = dfha4.merge(ids)
        else:
            dfhq1 = dfq1
            dfhq2 = dfq2
            dfhq3 = dfq3
            dfhq4 = dfq4
            dfha1 = dfa1
            dfha2 = dfa2
            dfha3 = dfa3
            dfha4 = dfa4
        out_q = [dfhq1, dfhq2, dfhq3, dfhq4]
        out_a = [dfha1, dfha2, dfha3, dfha4]
        # Print dataset size
        for h in range(4):
            print('Merged # of firm-year-analysts at quarter horizon ' + str(h+1) + ':', len(out_q[h]))
            print('Merged # of firm-years at quarter horizon ' + str(h+1) + ':', len(out_q[h].drop_duplicates(['i', 't'])))
            print('Merged # of firm-year-analysts at year horizon ' + str(h+1) + ':', len(out_a[h]))
            print('Merged # of firm-years at year horizon ' + str(h+1) + ':', len(out_a[h].drop_duplicates(['i', 't'])))
    else:
        # Get all dfs
        [dfq1, dfq2, dfq3, dfq4] = [make_df(estimator, 'q', h, W, scaled_p, iqr_drop, drop_winsor) for h in range(1,5)]
        [dfa1, dfa2, dfa3] = [make_df(estimator, 'a', h, W, scaled_p, iqr_drop, drop_winsor) for h in range(1,4)]
        # Match datasets
        if matchid != ['']:
            # Quartely
            dfhq1 = dfq1.merge(dfq2[matchid].drop_duplicates()).merge(dfq3[matchid].drop_duplicates()).merge(dfq4[matchid].drop_duplicates())
            dfhq2 = dfq2.merge(dfq1[matchid].drop_duplicates()).merge(dfq3[matchid].drop_duplicates()).merge(dfq4[matchid].drop_duplicates())
            dfhq3 = dfq3.merge(dfq1[matchid].drop_duplicates()).merge(dfq2[matchid].drop_duplicates()).merge(dfq4[matchid].drop_duplicates())
            dfhq4 = dfq4.merge(dfq1[matchid].drop_duplicates()).merge(dfq2[matchid].drop_duplicates()).merge(dfq3[matchid].drop_duplicates())
            # Annual
            dfha1 = dfa1.merge(dfa2[matchid].drop_duplicates()).merge(dfa3[matchid].drop_duplicates())
            dfha2 = dfa2.merge(dfa1[matchid].drop_duplicates()).merge(dfa3[matchid].drop_duplicates())
            dfha3 = dfa3.merge(dfa1[matchid].drop_duplicates()).merge(dfa2[matchid].drop_duplicates())
            if across_freq: # Merge annual and quarterly togethers
                acrossid = [x if x != 't' else 'year' for x in matchid]
                ids = dfha1[acrossid].drop_duplicates().merge(dfhq1[acrossid].drop_duplicates())
                dfhq1 = dfhq1.merge(ids)
                dfhq2 = dfhq2.merge(ids)
                dfhq3 = dfhq3.merge(ids)
                dfhq4 = dfhq4.merge(ids)
                dfha1 = dfha1.merge(ids)
                dfha2 = dfha2.merge(ids)
                dfha3 = dfha3.merge(ids)
        else:
            dfhq1 = dfq1
            dfhq2 = dfq2
            dfhq3 = dfq3
            dfhq4 = dfq4
            dfha1 = dfa1
            dfha2 = dfa2
            dfha3 = dfa3
        out_q = [dfhq1, dfhq2, dfhq3, dfhq4]
        out_a = [dfha1, dfha2, dfha3]
        # Print dataset size
        for h in range(4):
            print('Merged # of firm-year-analysts at quarter horizon ' + str(h+1) + ':', len(out_q[h]))
            print('Merged # of firm-years at quarter horizon ' + str(h+1) + ':', len(out_q[h].drop_duplicates(['i', 't'])))
        for h in range(3):
            print('Merged # of firm-year-analysts at year horizon ' + str(h+1) + ':', len(out_a[h]))
            print('Merged # of firm-years at year horizon ' + str(h+1) + ':', len(out_a[h].drop_duplicates(['i', 't'])))
    return [out_q, out_a]


##############################################################################
## FUNCTIONS FOR GMM OF STRUCTURAL MODEL
##############################################################################

# Score function for GMM estimation with Y = 1 and sigma_xi = 0
def h_Y1(X, theta):
    # Organize data and parameters
    splits = [1,2,3]
    nW = np.shape(X)[1] - 5
    splits = splits + [splits[-1] + nW, splits[-1] + nW + 1]
    EF, FFj, FFjk, W, j, k = np.hsplit(X, splits)  # Column vectors (except W)
    Sigma, alpha, *gamma_z = theta
    gW = (W @ gamma_z)[:, None]                    # Ensure column vector
    jk = (j == k).astype(int)
    # Form residuals
    eps1 = jk*EF - alpha*gW
    eps2 = jk*FFj - (alpha**2)*gW - Sigma
    eps3 = (1-jk)*FFjk - (alpha**2)*gW
    # Form orthogonality conditions
    h1 = eps1 * W
    h2 = eps2 * W
    h3 = eps3 * W
    # Output = nobs x nmoments
    return np.hstack([h1, h2, h3])

# Jacobian for GMM estimation with Y = 1 and xigma_xi = 0
def Dh_Y1(X, theta):
    # Organize data and parameters
    splits = [1,2,3]
    nW = np.shape(X)[1] - 5
    splits = splits + [splits[-1] + nW, splits[-1] + nW + 1]
    EF, FFj, FFjk, W, j, k = np.hsplit(X, splits)  # Column vectors (except W)
    Sigma, alpha, *gamma_z = theta
    gW = (W @ gamma_z)[:, None]                    # Ensure column vector
    # Form Jacobian residuals
    N = np.shape(X)[0] # number of observations
    k = 3*nW # number of moments
    p = len(theta) # number of parameters
    Nzeros = np.zeros((N,1))
    Nones = np.ones((N,1))
    Dh_dSigma = np.hstack([Nzeros, -Nones, Nzeros])
    Dh_dalpha = np.hstack([-gW, -2*alpha*gW, -2*alpha*gW])
    Dh_dgamma_z1 = np.hstack([-alpha*Nones, -(alpha**2)*Nones, -(alpha**2)*Nones])
    Dh_dgamma_zs = [np.multiply(Dh_dgamma_z1, W[:,i].reshape(-1,1)) for i in range(nW)] # list because multiple gamma_z's
    residuals = [Dh_dSigma, Dh_dalpha] +  Dh_dgamma_zs
    # Product of residuals with instruments
    Dh = np.zeros((N, p, k))
    for pi in range(p):
        Dh[:,pi,:] = (residuals[pi][...,None]*W[:,None]).reshape(N, -1)
    # Output = nobs x nparams x nmoments (G' for each observation in usual GMM notation)
    return Dh

# GMM estimation with Y = 1
def GMM_estimation(df, min_j, normalize_mse=True, variance=False, normalizer=None):
    # Filter based on number of analysts per i,t and create index
    N_it = df.groupby(['i','t'])['Y1'].count().reset_index()
    N_it = N_it[N_it.Y1 >= min_j]
    df = df.merge(N_it[['i','t']])
    print('Final sample size after filtering on # analysts per it:', len(df))
    # Get list of Ws
    Ws = [x for x in df.columns if 'W' in x]
    df_it = df[['i','t', 'delta', 'E_it'] + Ws].drop_duplicates(['i','t'])
    print('Final # of firm-quarters:', len(df_it))
    # Counter number of analyst level interactions
    N_it['nC2'] = N_it['Y1'].apply(nC2)
    N_it = N_it.drop(columns = 'Y1').merge(df_it).sort_values(['i', 't']).reset_index()
    # Make dataset with all two way interactions of analyst forecasts
    N_it_Ws = N_it[Ws].to_numpy()
    df = df.merge(N_it[['index', 'i', 't']]).sort_values(['i', 't', 'j'])
    Fijt = df[['index', 'j', 'F*_ijt']].to_numpy() # index = i,t unique
    Y2 = np.zeros(N_it.nC2.sum())
    indexit = np.zeros(N_it.nC2.sum())
    id1 = np.zeros(N_it.nC2.sum())
    id2 = np.zeros(N_it.nC2.sum())
    X2 = np.zeros((N_it.nC2.sum(), len(Ws)))
    IT = len(N_it)
    starts = np.array([np.sum(N_it.nC2[:i]) for i in range(IT)])
    ends = np.array([np.sum(N_it.nC2[:i+1]) for i in range(IT)])
    for it in range(IT):
        X2[starts[it]:ends[it],:] = N_it_Ws[it, :]
        fcasts = Fijt[Fijt[:,0] == it, 2]
        analysts = Fijt[Fijt[:,0] == it, 1]
        [Y2[starts[it]:ends[it]],
            id1[starts[it]:ends[it]],
            id2[starts[it]:ends[it]]] = multiply_nC2(fcasts, analysts)
        indexit[starts[it]:ends[it]] = it
    dfdict = {'F*_ijtF*_ikt':Y2, 'index':indexit, 'j':id1, 'k':id2}
    for ix in range(len(Ws)):
        dfdict[Ws[ix]] = X2[:, ix]
    reg2df = pd.DataFrame(dfdict)
    # Stack in observations where j = k
    df['k'] = df['j']
    df['F*_ijtF*_ikt'] = df['F*_ijt^2']
    Fijt_sq = df[['F*_ijtF*_ikt', 'index', 'j', 'k'] + Ws].copy()
    gmmdata = pd.concat([reg2df, Fijt_sq]).sort_values(['index', 'j', 'k'])
    # Merge in E*F* ON J
    gmmdata = gmmdata.merge(df[['index', 'j', 'F*_ijtE*_it']], on = ['index', 'j'])
    # Scale columns to account for weighting
    gmmdata['F*_ijtE*_it'] = gmmdata['F*_ijtE*_it']*len(gmmdata)/len(df)
    gmmdata['F*_ijtF*_ikt_j=k'] = gmmdata['F*_ijtF*_ikt']*len(gmmdata)/len(df)
    gmmdata['F*_ijtF*_ikt_j!=k'] = gmmdata['F*_ijtF*_ikt']*len(gmmdata)/(len(gmmdata) - len(df))
    # Reorganize columns to match GMM scores and convert to array
    varlist = ['F*_ijtE*_it', 'F*_ijtF*_ikt_j=k', 'F*_ijtF*_ikt_j!=k'] + Ws + ['j', 'k']
    gmmarray = gmmdata[varlist].to_numpy()
    # Run GMM with bootstrap if requested
    init = [0.01, 1] + [0] * len(Ws)
    if global_bootstrap:
        bootdata = bootstrap_data(
            gmmdata.merge(df[['index', boot_id]].drop_duplicates(), on = 'index')[varlist + [boot_id]], 
            Nb, boot_id
            )
        bootdata = [d[varlist].to_numpy() for d in bootdata]
        gmmres = fgmm.gmm(h_Y1, gmmarray, init, iterated = True, bootdata = bootdata, N_boot = Nb)
    else:
        gmmres = fgmm.gmm(h_Y1, gmmarray, init, scorederiv = Dh_Y1, iterated = True)
    estimate = gmmres[1]
    Sigma = estimate[0]
    alpha = estimate[1]
    gamma_z = estimate[2:]
    # Calculate theta and delta
    theta = 0
    for ix in range(len(Ws)):
        theta = theta + gamma_z[ix]*np.mean(df[Ws[ix]])
    delta = df['delta'].mean()
    # Normalizations - if you normalized by price, it will be taken into account here!
    variance_eps = df['E_it'].var()
    squared_eps = (df['E_it']**2).mean()
    if normalizer is None:
        if normalize_mse == True:
            if variance == True:
                norm = variance_eps / 100
            else:
                norm = squared_eps / 100
        else:
            norm = 1
    else:
        norm = normalizer
    theta = theta / norm
    delta = delta / norm
    Sigma = Sigma / norm
    params = {'Sigma':Sigma, 'Theta':theta, 'alpha':alpha,'Delta':delta, 'gamma_z':gamma_z}
    params['Delta_p'] = (1 - params['alpha'])**2 * params['Theta']
    # Calculate standard errors on parameters using delta method for Theta
    se_params = {'Sigma':gmmres[2][0]/norm, 'Theta':-99, 'alpha':gmmres[2][1], 'Delta':-99, 'gamma_z':gmmres[2][2:]}
    df_it = df.drop_duplicates(['i', 't']).copy()
    if global_bootstrap:
        se_params['Delta'] = np.std([np.mean(bootstrap_single(df, boot_id, rs = i)['delta']) for i in range(Nb)]) / norm
    else:
        se_params['Delta'] = df_it['delta'].std() / np.sqrt(len(df_it)) / norm
    VCV = []
    for ix in range(len(Ws)):
        VCV.append(se_params['gamma_z'][ix]**2 * len(df))
    for ix in range(1,len(Ws)):
        VCV.append(df_it[Ws[ix]].var())
    G = []
    for ix in range(len(Ws)):
        G.append(np.mean(df[Ws[ix]]))
    for ix in range(1,len(Ws)):
        G.append(gamma_z[ix])
    VCV = np.diag(VCV)
    G = np.array(G)
    se_params['Theta'] = np.sqrt(G @ VCV @ G / len(df_it)) / norm
    # Calculate standard errors for Delta_p based on delta method
    VCV = np.diag([se_params['alpha']**2 * len(df), (se_params['Theta'] * norm)**2 * len(df)])
    G = [-2 * (1 - params['alpha']) * (params['Theta'] * norm), (1 - params['alpha'])**2 ]
    se_params['Delta_p'] = np.sqrt(G @ VCV @ G / len(df_it)) / norm
    return [gmmres, params, se_params, norm]

# Multi-horizon GMM estimation
def run_GMMs_byh(matchid, across_freq, estimator, W, scaled_p, include_4, iqr_drop=5, drop_winsor=True, size_half=-1):
    print('-------------------------------------------------------------------------------------',
        '\nML Estimator for g():', estimator, '\nMatching ID:', matchid, '\nMerge Across Frequences:', across_freq,
              '\nWs:', W, '\nIQRs for Winsorization:', iqr_drop, '\nInclude 4 year horizon:', include_4)
    # Get datasets
    [dfs_q, dfs_a] = make_dfs_matched(matchid, across_freq, estimator, W, include_4, scaled_p, iqr_drop, drop_winsor)
    dfs = dfs_q + dfs_a
    # Split by size if requested
    if size_half > 0:
        for i in range(len(dfs)):
            med = dfs[i]['LOG_AT'].median()
            if size_half == 2:
                dfs[i] = dfs[i][dfs[i]['LOG_AT'] > med].copy()
            elif size_half == 1:
                dfs[i] = dfs[i][dfs[i]['LOG_AT'] <= med].copy()
    # Form lists
    gmmres, Sigma, Theta, Delta, alpha, Delta_p, se_Sigma, se_Theta, se_Delta, se_alpha, se_Delta_p, norms, var_eps = [[] for _ in range(13)]
    rho_z, rho_eta, rho_eps = [[0] for _ in range(3)]
    # Loop over horizons
    for i in range(len(dfs)):
        # Run GMM by horizon and store results
        df = dfs[i]
        out = GMM_estimation(df, 0, variance = True)
        gmmres.append(out[0])
        Sigma.append(out[1]['Sigma'])
        Theta.append(out[1]['Theta'])
        Delta.append(out[1]['Delta'])
        alpha.append(out[1]['alpha'])
        Delta_p.append(out[1]['Delta_p'])
        se_Sigma.append(out[2]['Sigma'])
        se_Theta.append(out[2]['Theta'])
        se_Delta.append(out[2]['Delta'])
        se_alpha.append(out[2]['alpha'])
        se_Delta_p.append(out[2]['Delta_p'])
        norms.append(out[3])
        # Calculate auxliary parameters
        var_eps.append((np.var(df['E*_it']) - Theta[i] * norms[i]) / norms[i])
        if i > 0 and across_freq and matchid == ['i','t','j']:
            rho_z.append(np.cov(df['F*_ijt'], dfs[i-1]['E*_it'])[0,1] / alpha[i] / (Theta[i-1] * norms[i-1]))
            rho_eps.append(np.cov(df['E*_it'], dfs[i-1]['E*_it'])[0,1] / (var_eps[i-1] * norms[i-1]) - rho_z[i] * (Theta[i-1] * norms[i-1]) / (var_eps[i-1] * norms[i-1]))
            rho_eta.append(np.cov(df['F*_ijt'], dfs[i-1]['F*_ijt'])[0,1] / (Sigma[i-1] * norms[i-1]) - alpha[i] * alpha[i-1] * rho_z[i] * Theta[i-1] / Sigma[i-1])
    # Plot results
    if include_4:
        quarters = [1, 2, 3, 4, 4, 8, 12, 16]
    else:
        quarters = [1, 2, 3, 4, 4, 8, 12]
    quarters_txt = [x/4 for x in quarters[4:]]
    fig, axes = plt.subplots(2,2)
    for i in range(len(quarters)):
        if i <= 3:
            col = 'maroon'
            mk = 'o'
        else:
            col = 'darkblue'
            mk = 's'
        axes[0,0].errorbar(quarters[i], Sigma[i], yerr = [1.96*x for x in se_Sigma][i], color = col, capsize = 4, fmt = mk, alpha = 0.7)
        axes[0,0].set_title("Noise: $\Sigma_h$")
        axes[1,0].errorbar(quarters[i], Delta[i], yerr = [1.96*x for x in se_Delta][i], color = col, capsize = 4, fmt = mk, alpha = 0.7)
        axes[1,0].set_title("Public Bias: $\Delta_h$")
        axes[0,1].errorbar(quarters[i], Theta[i], yerr = [1.96*x for x in se_Theta][i], color = col, capsize = 4, fmt = mk, alpha = 0.7)
        axes[0,1].set_title("Soft Info: $\Theta_h$")
        axes[1,1].errorbar(quarters[i], alpha[i], yerr = [1.96*x for x in se_alpha][i], color = col, capsize = 4, fmt = mk, alpha = 0.7)
        axes[1,1].set_title("Weight on Soft Info: $\\alpha_h$")
    for i in range(2):
        for j in range(2):
            axes[i,j].set_xticks(quarters[4:])
            axes[i,j].set_xticklabels(quarters_txt)
            axes[i,j].set_xlabel('Forecast Horizon: $h$ years')
            if not (i == 1 and j == 1):
                axes[i,j].yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.subplots_adjust(hspace = 0.6, wspace = 0.4)
    fig.set_size_inches(8, 8)
    os.chdir(figdir)
    figfile = "GMMtermstructure_match" + ''.join(matchid) + '_' + estimator
    if scaled_p:
        figfile += '_scaledp'
    if across_freq:
        figfile += '_acrossfreq'
    if include_4:
        figfile += '_include4'
    if W[0] != '':
        figfile = figfile + '_' + ''.join(W)
    if size_half > 0:
        figfile = figfile + '_size' + str(size_half)
    plt.savefig(figfile + '.pdf', bbox_inches = 'tight')
    # Plot comparison of bias on public and soft information
    fig, axes = plt.subplots(1,1)
    axes.errorbar(quarters, Delta, yerr = [1.96*x for x in se_Delta], color = 'maroon', capsize = 4, fmt = 'o', alpha = 0.7, label = "Public Bias: $\Delta_h$")
    axes.errorbar(quarters, Delta_p, yerr = [1.96*x for x in se_Delta_p], color = 'darkblue', capsize = 4, fmt = 's', alpha = 0.7, label = "Soft Bias: $(1-\\alpha_h)^2\Theta_h$")
    axes.legend()
    axes.set_xticks(quarters[4:])
    axes.set_xticklabels(quarters_txt)
    axes.set_xlabel('Forecast Horizon: $h$ years')
    axes.yaxis.set_major_formatter(mtick.PercentFormatter())
    os.chdir(figdir)
    plt.savefig(figfile.replace('termstructure', 'biascompare') + '.pdf', bbox_inches = 'tight')
    # Form output of annual dataset
    annids = range(len(dfs_q), len(dfs))
    pout_a = pd.DataFrame({
            'alpha':alpha[4:],
            'se_alpha':se_alpha[4:],
            'Delta':[Delta[i]*norms[i] for i in annids],
            'se_Delta':[se_Delta[i]*norms[i] for i in annids],
            'Theta':[Theta[i]*norms[i] for i in annids],
            'se_Theta':[se_Theta[i]*norms[i] for i in annids],
            'Sigma':[Sigma[i]*norms[i] for i in annids],
            'se_Sigma':[se_Sigma[i]*norms[i] for i in annids],
            'Delta_p':[Delta_p[i]*norms[i] for i in annids],
            'se_Delta_p':[se_Delta_p[i]*norms[i] for i in annids],
            'norms':norms[4:],
            'Jstat':[x[3] for x in gmmres[4:]]
        }, index = list(range(len(dfs_a))))
    # Form quarterly and annual parameters
    pout = pd.DataFrame({
            'alpha':alpha,
            'se_alpha':se_alpha,
            'Delta':[Delta[i]*norms[i] for i in range(len(dfs))],
            'se_Delta':[se_Delta[i]*norms[i] for i in range(len(dfs))],
            'Theta':[Theta[i]*norms[i] for i in range(len(dfs))],
            'se_Theta':[se_Theta[i]*norms[i] for i in range(len(dfs))],
            'Sigma':[Sigma[i]*norms[i] for i in range(len(dfs))],
            'se_Sigma':[se_Sigma[i]*norms[i] for i in range(len(dfs))],
            'Delta_p':[Delta_p[i]*norms[i] for i in range(len(dfs))],
            'se_Delta_p':[se_Delta_p[i]*norms[i] for i in range(len(dfs))],
            'norms':norms,
            'Jstat':[x[3] for x in gmmres]
        }, index = list(range(len(dfs))))
    # Pickle
    os.chdir(outputdir)
    output = [pout_a, dfs_a, pout, dfs_q]
    file = open(figfile.replace('termstructure', '') + '.pcl', 'wb')
    pickle.dump(output, file)
    file.close()
    return output

# Function to read back in GMM results
def load_GMMs_byh(matchid, across_freq, estimator, W, scaled_p, include_4):
    figfile = "GMM_match" + ''.join(matchid) + '_' + estimator
    if scaled_p:
        figfile += '_scaledp'
    if across_freq:
        figfile += '_acrossfreq'
    if include_4:
        figfile += '_include4'
    if W[0] != '':
        figfile = figfile + '_' + ''.join(W)
    os.chdir(outputdir)
    file = open(figfile + '.pcl', 'rb')
    output = pickle.load(file)
    file.close()
    return output

# Function to output results from GMM in MSE decomposition table
def make_decomp_table(matchid, across_freq, estimator, W, scaled_p, include_4):
    # Loop through different list of controls
    [dfs_a, pout, dfs_q] = load_GMMs_byh(matchid, across_freq, estimator, W, scaled_p, include_4)[1:]
    # Calculate MSEa-MSEm for each horizon and append
    MSEdiffs = []; MSEdiffs_aa = []
    countJ = []
    for df in dfs_q + dfs_a:
        df_it = df.groupby(['i', 't'])[['E_it', 'F_ijt', 'Fe_it']].mean()
        ## itj unique
        df_itj = df[['i', 't', 'j']].merge(df_it, how = 'left', on = ['i', 't'])
        df_itj['se_a'] = (df_itj['E_it'] - df_itj['F_ijt'])**2
        df_itj['se_m'] = (df_itj['E_it'] - df_itj['Fe_it'])**2
        df_itj = df_itj.merge(df[['i', 't', 'j', 'F_ijt']].rename(columns = {'F_ijt':'F_ijt_ind'}), 
                              on = ['i', 'j', 't'], how = 'left')
        df_itj['se_aa'] = (df_itj['E_it'] - df_itj['F_ijt_ind'])**2
        df_itj = clip_outlier_iqr(df_itj, ['se_a', 'se_m'], 5)
        df_itj = clip_outlier_iqr(df_itj, ['se_aa'], 5)
        MSEa = np.mean(df_itj['se_a'])
        MSEm = np.mean(df_itj['se_m'])
        MSEaa = np.mean(df_itj['se_aa'])
        EJinv = (1/df.groupby(['i', 't'])['j'].transform('count')).mean()
        ## Append
        MSEdiffs.append(MSEa - MSEm)
        MSEdiffs_aa.append(MSEaa - MSEm)
        countJ.append(EJinv)
    pout['MSE_diff'] = MSEdiffs
    pout['EJinv'] = countJ
    # Calculate David statistic
    pout['MSE_diff_aa'] = MSEdiffs_aa
    pout['Sigma_check'] = (pout['MSE_diff_aa'] - pout['MSE_diff'])/(1 - pout['EJinv'])
    pout['crowd_gain'] = (pout['MSE_diff_aa'] - pout['MSE_diff'])/np.abs(pout['MSE_diff_aa'])
    # Calculate model implied difference in MSEs and error with actual data
    decomp = pout[['Theta', 'Sigma', 'Delta', 'Delta_p', 'MSE_diff', 'EJinv', 'norms']].copy()
    decomp['MSEdiffmodel'] = decomp['Delta'] + decomp['Delta_p'] + decomp['EJinv']*decomp['Sigma'] - decomp['Theta']
    decomp['error_p'] = (decomp['MSEdiffmodel'] - decomp['MSE_diff'])/np.abs(decomp['MSE_diff'])
    # Normalize each element of decomposition by GMM normalizers
    decomp['MSEdiffmodel'] = decomp['MSEdiffmodel'] / decomp['norms']
    decomp['Public Bias'] = decomp['Delta'] / decomp['norms']
    decomp['Soft Bias'] = decomp['Delta_p'] / decomp['norms']
    decomp['Noise'] = decomp['EJinv'] * decomp['Sigma'] / decomp['norms']
    decomp['Soft Information'] = -decomp['Theta'] / decomp['norms']
    # Format
    decomp = decomp.applymap(float).round(2).applymap(str)
    decomp[['MSEdiffmodel', 'Soft Information', 'Public Bias', 'Soft Bias', 'Noise']] += '%'
    hs = ['1 Quarters', '2 Quarters', '3 Quarters', '4 Quarters', '1 Years', '2 Years', '3 Years']
    if include_4:
        hs += ['4 Years']
    decomp['Horizon'] = hs
    # Tabulate
    table = decomp[['Horizon', 'MSEdiffmodel', 'Soft Information', 'Public Bias', 'Soft Bias', 'Noise']].copy()
    tex = tabulate(table, showindex = False, tablefmt = 'latex_booktabs', headers = 'keys', stralign = 'center', numalign = 'center')
    tex = tex.replace('tabular}{c', 'tabular}{l')
    tex = tex.replace('Horizon', 'Horizon: $h$')
    tex = tex.replace('MSEdiffmodel', '$MSE^a - MSE^e$')
    tex = tex.replace('Soft Information', '$-\Theta$')
    tex = tex.replace('Soft Bias', '$(1-\\alpha)^2\Theta$')
    tex = tex.replace('Public Bias', '$\Delta$')
    tex = tex.replace('Noise', '$\\frac{1}{J}\Sigma$')
    # Output
    figfile = "GMMdecomp_match" + ''.join(matchid) + '_' + estimator
    if scaled_p:
        figfile += '_scaledp'
    if across_freq:
        figfile += '_acrossfreq'
    if include_4:
        figfile += '_include4'
    if W[0] != '':
        figfile = figfile + '_' + ''.join(W)
    os.chdir(outputdir)
    with open(figfile + '.tex','w') as output:
        output.write(tex)



##############################################################################
## FUNCTIONS FOR ESTIMATION OF PT MODEL
##############################################################################

# Minimum distance estimation of PT model
def PT_estimation(inc_beta0, matchid, across_freq, estimator, W, scaled_p, include_4):
    # Run first-stage GMM and get results
    [gmm1, dfs] = load_GMMs_byh(matchid, across_freq, estimator, W, scaled_p, include_4)[:2]
    # Number of horizons for MDE
    if include_4:
        Nh = 4
    else:
        Nh = 3
    # Get data moments needed for MDE
    ## Our moments from before
    alphah = [gmm1.loc[h, 'alpha'] for h in range(Nh)]
    Deltah = [gmm1.loc[h, 'Delta'] for h in range(Nh)]
    Sigmah = [gmm1.loc[h, 'Sigma'] for h in range(Nh)]
    Thetah = [gmm1.loc[h, 'Theta'] for h in range(Nh)]
    se_alphah = [gmm1.loc[h, 'se_alpha'] for h in range(Nh)]
    se_Deltah = [gmm1.loc[h, 'se_Delta'] for h in range(Nh)]
    se_Sigmah = [gmm1.loc[h, 'se_Sigma'] for h in range(Nh)]
    ## Regression of F on E|X with constant
    regFEh = [run_OLS(dfs[h], ['Fe_it'], 'F_ijt') for h in range(Nh)]
    slopeFEh = [x.loc['Fe_it', 'coef'] for x in regFEh]
    consFEh = [x.loc['Intercept', 'coef'] for x in regFEh]
    if global_bootstrap:
        se_slopeFEh = []; se_consFEh = []
        for df in dfs:
            bs_slope = []; bs_const = []
            for i in range(Nb):
                ols_bs = run_OLS(bootstrap_single(df, 'i', rs = i), ['Fe_it'], 'F_ijt')
                bs_slope.append(ols_bs.loc['Fe_it', 'coef'])
                bs_const.append(ols_bs.loc['Intercept', 'coef'])
            se_slopeFEh.append(np.std(bs_slope))
            se_consFEh.append(np.std(bs_const))
    else:
        se_slopeFEh = [x.loc['Fe_it', 'se'] for x in regFEh]
        se_consFEh = [x.loc['Intercept', 'se'] for x in regFEh]
    ## Additional moments
    MSEmh = [np.mean((df['Fe_it'] - df['E_it'])**2) for df in dfs]
    Ex2h = [np.mean(df['Fe_it']**2) for df in dfs]
    Exh = [np.mean(df['Fe_it']) for df in dfs]
    # Moment vector and standard errors in data
    m_data = []
    se_data = []
    for h in range(Nh):
        m_data += [alphah[h], Deltah[h], Sigmah[h]]
        se_data += [se_alphah[h], se_Deltah[h], se_Sigmah[h]]
        if inc_beta0:
            m_data += [slopeFEh[h], consFEh[h]]
            se_data += [se_slopeFEh[h], se_consFEh[h]]
    m_data = np.array(m_data)
    se_data = np.array(se_data)*1.96
    # Function for mh a la Patton-Timmermann
    def m_h_PT(kappa, h):
        return kappa**2 / (kappa**2 + MSEmh[h] - Thetah[h])
    # Moment vector for MDE of PT model
    if inc_beta0:
        Nm = 5*Nh
        def m_PT(theta):
            # Extract parameters
            beta_0, beta_x, beta_z, sigma2_nu, kappa = theta
            mh = [m_h_PT(kappa, h) for h in range(Nh)]
            # Form moment conditions for each horizon
            g = []
            for h in range(Nh):
                g += [
                    (1-mh[h])*beta_z + mh[h],
                    ((1-mh[h])**2)*(beta_0**2 + beta_0*(beta_x-1)*Exh[h] + ((beta_x-1)**2)*Ex2h[h]),
                    ((1-mh[h])**2)*sigma2_nu,
                    (1-mh[h])*beta_x + mh[h],
                    (1-mh[h])*beta_0
                ]
            return np.array(g)
    else:
        Nm = 3*Nh
        def m_PT(theta):
            # Extract parameters
            beta_x, beta_z, sigma2_nu, kappa = theta
            mh = [m_h_PT(kappa, h) for h in range(Nh)]
            # Form moment conditions for each horizon
            g = []
            for h in range(Nh):
                g += [
                    (1-mh[h])*beta_z + mh[h],
                    ((1-mh[h])**2)*((1-beta_x)**2)*Ex2h[h],
                    ((1-mh[h])**2)*sigma2_nu
                ]
            return np.array(g)
    N_byh = int(Nm/Nh)
    # Construct different weighting matrices for MDE
    O_scaled = [] # scaled moments matrix
    O_diag = [] # inverse diagonal matrix
    for h in range(Nh):
        O_scaled += [alphah[h]**(-2), Deltah[h]**(-2), Sigmah[h]**(-2)]
        O_diag += [se_alphah[h]**(-2), se_Deltah[h]**(-2), se_Sigmah[h]**(-2)]
        if inc_beta0:
            O_scaled += [slopeFEh[h]**(-2), consFEh[h]**(-2)]
            O_diag += [se_slopeFEh[h]**(-2), se_consFEh[h]**(-2)]
    O_scaled = np.diag(O_scaled)
    O_diag = np.diag(O_diag)
    # Function to minimize
    def obj_mde(theta, Omega):
        if theta[-1] < 0:
            return 10e10
        else:
            g = m_PT(theta) - m_data
            return g @ Omega @ g
    # David procedure to get initial estimate
    def obj_mde_noise(theta):
        if theta[-1] < 0:
            return 10e10
        else:
            theta1 = np.append([0,0], theta)
            if inc_beta0:
                theta1 = np.append([0], theta1)
            gs = m_PT(theta1)
            g = gs[[2 + N_byh*pi for pi in range(Nh)]] - m_data[[2 + N_byh*pi for pi in range(Nh)]]
        return g @ g
    mde0 = basinhopping(func = obj_mde_noise, x0 = [Sigmah[0], np.sqrt(MSEmh[0] - Thetah[0])], niter = 1000, seed = 12345)
    theta0 = mde0.x
    beta_zs = np.array([])
    beta_xs = np.array([])
    for h in range(Nh):
        beta_zs = np.append(beta_zs, (alphah[h] - m_h_PT(theta0[1], h)) / (1 - m_h_PT(theta0[1], h)))
        beta_xs = np.append(beta_xs, 1 - np.sqrt(Deltah[h] / ((1 - m_h_PT(theta0[1], h))**2) / Ex2h[h]))
    theta0 = np.append([np.mean(beta_xs), np.mean(beta_zs)], theta0)
    if inc_beta0:
        theta0 = np.append(0, theta0)
    # Run MDE
    theta00 = [1,1] + [Sigmah[0], np.sqrt(MSEmh[0] - Thetah[0])]
    if inc_beta0:
        theta00 = [0] + theta00
    mde = basinhopping(func = obj_mde, x0 = theta00, minimizer_kwargs = {'args':(O_diag)}, niter = 1000, seed = 12345)
    thetahat = mde.x
    # Plot data versus model results
    horizons = list(range(1,Nh+1))
    fig = plt.figure(figsize = (12,3))
    gs = gridspec.GridSpec(nrows = 1, ncols = 3, wspace = 0.4, hspace = 0.6)
    titles = ["Soft Bias: $\\alpha_h$", "Public Bias: $\Delta_h$", "Noise: $\Sigma_h$", "Slope Coefficient: $\delta^1_h$", "Intercept: $\delta^0_h$"]
    # for i in range(N_byh):
    for i in range(3):
        pid = i + np.array([N_byh*pi for pi in range(Nh)])
        if i < 3:
            axes = fig.add_subplot(gs[0,i])
        else:
            axes = fig.add_subplot(gs[1,i-3])
        if i >= 1 and i < 3:
            norm = gmm1['norms'].to_numpy()
            axes.yaxis.set_major_formatter(mtick.PercentFormatter())
        else:
            norm = np.ones(Nh)
        datai = np.divide(m_data[pid], norm)
        modeli = np.divide(m_PT(thetahat)[pid], norm)
        sedatai = np.divide(se_data[pid], norm)
        axes.errorbar(horizons, datai, yerr = sedatai, label = 'Data (95% CI)', color = 'maroon', capsize = 4, fmt = 'o', alpha = 0.7)
        axes.plot(horizons, modeli, 'o', label = 'Model', color = 'darkblue', alpha = 0.7)
        axes.set_xticks(horizons)
        axes.set_title(titles[i])
        axes.set_xlabel('Forecast Horizon: $h$ years')
        if i == 0 :
            axes.legend()
    os.chdir(figdir)
    figfile = "PTestimation_match" + ''.join(matchid) + '_' + estimator
    if inc_beta0:
        figfile += '_beta0'
    if scaled_p:
        figfile += '_scaledp'
    if across_freq:
        figfile += '_acrossfreq'
    if include_4:
        figfile += '_include4'
    if W[0] != '':
        figfile = figfile + '_' + ''.join(W)
    plt.savefig(figfile + '.pdf', bbox_inches = 'tight')
    # Plot implied mh's
    fig, axes = plt.subplots(1,1)
    implied_mhs = [m_h_PT(mde.x[-1],h) for h in range(Nh)]
    axes.bar(horizons, implied_mhs, color = 'darkblue', label = 'Model Implied $m_h$')
    axes.legend()
    axes.set_ylim([0,1])
    axes.set_xticks(horizons)
    axes.set_xlabel('Forecast Horizon: $h$ years')
    plt.tight_layout()
    plt.savefig(figfile + '_mh.pdf', bbox_inches = 'tight')
    # Calculate standard errors of estimates
    thetas_se = []
    h_diff = 0.01
    for i in range(len(thetahat)):
        thetase = thetahat.copy()
        thetase[i] = thetahat[i]*(1 + h_diff)
        thetas_se.append(thetase.copy())
        thetase[i] = thetahat[i]*(1 - h_diff)
        thetas_se.append(thetase.copy())
    G = np.zeros((Nm, len(thetahat)))
    for i in range(Nm):
        for j in range(len(thetahat)):
            G[i,j] = (m_PT(thetas_se[2*j])[i] - m_PT(thetas_se[2*j+1])[i]) / (thetas_se[2*j][j] - thetas_se[2*j+1][j])
    Chunk = G.T @ O_diag @ G
    ChunkInv = np.linalg.pinv(Chunk)
    XXX = ChunkInv @ G.T
    Omega = np.linalg.pinv(O_diag)
    V = XXX @ O_diag @ Omega @ O_diag.T @ XXX.T
    ses = np.sqrt(np.diag(V))
    # Output parameter estimates
    os.chdir(outputdir)
    output = [thetahat, ses, gmm1, dfs]
    # output = [thetahat, gmm1, dfs, ses]
    file = open(figfile + '.pcl', 'wb')
    pickle.dump(output, file)
    file.close()



##############################################################################
## FUNCTIONS FOR ADDITIONAL MODEL APPLICAITONS
##############################################################################

# Function to load PT estimation results
def load_PT(inc_beta0, matchid, across_freq, estimator, W, scaled_p, include_4):
    # Filename to read in
    figfile = "PTestimation_match" + ''.join(matchid) + '_' + estimator
    if inc_beta0:
        figfile += '_beta0'
    if scaled_p:
        figfile += '_scaledp'
    if across_freq:
        figfile += '_acrossfreq'
    if include_4:
        figfile += '_include4'
        Nh = 4
    else:
        Nh = 3
    if W[0] != '':
        figfile = figfile + '_' + ''.join(W)
    # Read in file from PT estimation
    os.chdir(outputdir)
    file = open(figfile + '.pcl', 'rb')
    [thetahat, ses, gmm1, dfs] = pickle.load(file)
    file.close()
    return [thetahat, gmm1, dfs, Nh, figfile]

# Function to examine cross-sectional relationship between volatilty and noise
def XS_noiseestimation(xsvar, nbins, xslab, inc_beta0, matchid, across_freq, estimator, W, scaled_p, include_4):
    # Read in results
    [thetahat, gmm1, dfs, Nh, figfile] = load_PT(inc_beta0, matchid, across_freq, estimator, W, scaled_p, include_4)
    norms = gmm1['norms'].to_numpy()
    # Extract estimated PT parameters
    sigma2_nu = thetahat[-2]
    kappa = thetahat[-1]
    # Loop to estimate and plot for each horizon
    fig, axes = plt.subplots(1,3)
    model = []
    data = []
    se_data = []
    for h in range(3):
        # Variable creation
        df = dfs[h]
        df['MSEm'] = (df['E_it'] - df['Fe_it'])**2
        # Merge in variable of interest
        os.chdir(datadir)
        dfit = pd.read_sas('datait_a' + str(h+1) + '.sas7bdat', format = 'sas7bdat', encoding = 'latin-1')
        df = df.merge(dfit[['gvkey', 'T_DATADATE', xsvar]], left_on = ['i', 't'], right_on = ['gvkey', 'T_DATADATE'])
        if nbins > 0:
            df = bin_data(remove_outlier(df, xsvar), xsvar, nbins)
            binvar = 'bin_' + xsvar
            bins = list(range(1,nbins+1))
            bin_means = df.groupby(binvar)[xsvar].mean()
        else:
            binvar = xsvar
            bins = df.groupby(binvar)[binvar].count() > 100
            bins = np.array(bins[bins].index)
        # Estimation by bins
        Sigmah_bins = []
        se_Sigmah_bins = []
        PTSigmah_bins = []
        for i in bins:
            dfi = df[df[binvar] == i].copy()
            out = GMM_estimation(dfi, 0, normalize_mse = False)
            Sigmah_bins.append(out[1]['Sigma'] / norms[h])
            se_Sigmah_bins.append(1.96 * out[2]['Sigma'] / norms[h])
            mhi = kappa**2 / (kappa**2 + dfi['MSEm'].mean() - out[1]['Theta'])
            PTSigmai = ((1-mhi)**2) * sigma2_nu / norms[h]
            PTSigmah_bins.append(PTSigmai)
        # Plot empirical noise across bins
        if nbins > 0:
            axes[h].errorbar(bin_means, Sigmah_bins, yerr = se_Sigmah_bins, color = 'darkblue', capsize = 4, fmt = 'o', alpha = 0.7)
            axes[h].yaxis.set_major_formatter(mtick.PercentFormatter())
            axes[h].set_title('$h$ = ' + str(h+1) + ' years')
        # Append
        data.append(Sigmah_bins)
        se_data.append(se_Sigmah_bins)
        model.append(PTSigmah_bins)
    # Clean up and output plot
    os.chdir(figdir)
    axes[0].set_ylabel('Estimated Noise: $\Sigma_h$')
    axes[1].set_xlabel(xslab)
    fig.set_size_inches(16,6)
    plt.savefig(figfile.replace('PTestimation', 'XSestimation_data_' + xsvar) + '.pdf', bbox_inches = 'tight')
    fig2, axes2 = plt.subplots(1,3)
    # Plot empirical noise vs. PT predicted noise
    for h in range(3):
        axes2[h].errorbar(model[h], data[h], yerr = se_data[h], color = 'darkblue', capsize = 4, fmt = 'o', alpha = 0.7)
        axes2[h].xaxis.set_major_formatter(mtick.PercentFormatter())
        axes2[h].yaxis.set_major_formatter(mtick.PercentFormatter())
        axes2[h].set_title('$h$ = ' + str(h+1) + ' years')
        x = np.linspace(*axes2[h].get_xlim())
        axes2[h].plot(x, x, color = 'black', ls = '--')
    axes2[0].set_ylabel('Estimated Noise: $\Sigma_h$')
    axes2[1].set_xlabel('Model Implied Noise: $\Sigma_h(\widehat{\kappa}, \widehat{\\sigma}_\\nu)$')
    fig2.set_size_inches(16,6)
    plt.savefig(figfile.replace('PTestimation', 'XSestimation_model_' + xsvar) + '.pdf', bbox_inches = 'tight')

# Calculate counter-factual CG coefficient
def CG_nonoise(inc_beta0, matchid, across_freq, estimator, W, scaled_p, include_4):
    # Read in results
    [thetahat, gmm1, dfs, Nh, figfile] = load_PT(inc_beta0, matchid, across_freq, estimator, W, scaled_p, include_4)
    # Extract estimated PT parameters
    if inc_beta0:
        beta_0, beta_x, beta_z, sigma2_nu, kappa = thetahat
    else:
        beta_x, beta_z, sigma2_nu, kappa = thetahat
    # Make merged dataset of each pair of horizons and calculate across-horizon statistics
    mergeid = ['i', 'j', 'LEAD_PENDS']
    mergecols = mergeid + ['t', 'F_ijt', 'F*_ijt', 'Fe_it', 'E_it']
    merge_dfs = []
    CGs = []; CGs_se = []
    CGcf = []
    os.chdir(datadir)
    for h in range(1,Nh):
        # Calculate price ratio across horizons
        data1 = pd.read_sas('datait_a' + str(h) + '.sas7bdat', format = 'sas7bdat', encoding = 'latin-1')
        data1.columns = map(str.upper, data1.columns)
        data2 = pd.read_sas('datait_a' + str(h+1) + '.sas7bdat', format = 'sas7bdat', encoding = 'latin-1')
        data2.columns = map(str.upper, data2.columns)
        prccs = data1[['GVKEY', 'LEAD_PENDS', 'PRCC']].merge(data2[['GVKEY', 'LEAD_PENDS', 'PRCC']], on = ['GVKEY', 'LEAD_PENDS'], suffixes = ('_1', '_2'))
        prccs['PRCC_ratio'] = prccs['PRCC_2'] / prccs['PRCC_1']
        # Merge forecasting datasets across horizons
        df12 = dfs[h].merge(dfs[h-1][mergecols], on = mergeid, suffixes = ('', '_lead'))
        # Adjust for price scaling differentials to match longer horizon
        df12 = df12.merge(prccs, left_on = ['i', 'LEAD_PENDS'], right_on = ['GVKEY', 'LEAD_PENDS'])
        leadcols = [x for x in df12.columns if 'lead' in x]
        df12[leadcols] = df12[leadcols].div(df12['PRCC_ratio'], axis = 'index')
        # Run CG regression
        df12['error'] = df12['E_it'] - df12['F_ijt_lead']
        df12['revision'] = df12['F_ijt_lead'] - df12['F_ijt']
        cg_reg = run_OLS(remove_outlier(df12, ['error', 'revision']), ['revision'], 'error')
        cg = cg_reg.loc['revision', 'coef']
        cg_se = cg_reg.loc['revision', 'se']
        # Estimate noises on merged dataset
        Sigmah = GMM_estimation(dfs[h-1].merge(df12[mergeid]), 0, normalize_mse=False)[1]['Sigma']
        Sigmah1 = GMM_estimation(dfs[h].merge(df12[mergeid]), 0, normalize_mse=False)[1]['Sigma']
        # Calculate counterfactual noise coefficient
        var_deltaF = (df12['F_ijt_lead'] - df12['F_ijt']).var()
        var_deltaFbar = var_deltaF - Sigmah - Sigmah1
        cg_cf = (cg * var_deltaF + Sigmah) / var_deltaFbar
        # Append
        merge_dfs.append(df12)
        CGs.append(cg)
        CGs_se.append(cg_se)
        CGcf.append(cg_cf)
    # Plot
    CGdf = pd.DataFrame({
        'Without Noise: $\\overline{\\beta}_{CG}$':CGcf,
        '+ Noise Attenuation Effect':[x * var_deltaFbar / var_deltaF for x in CGcf],
        '+ Noise Attenuation and Level Effects: $\\beta_{CG}$':CGs
    }, index = list(range(1,Nh)))
    fig, axes = plt.subplots(1,1)
    CGdf.plot(ax = axes, kind = 'bar', color=['skyblue', 'darkblue', 'maroon'])
    axes.legend()
    axes.set_xlabel('Forecast Horizon: $h$ years')
    axes.set_ylabel('CG Coefficient')
    plt.xticks(rotation = 0)
    plt.tight_layout()
    os.chdir(figdir)
    plt.savefig(figfile.replace('PTestimation', 'CGnonoise') + '.pdf', bbox_inches = 'tight')


##############################################################################
## MAIN SCRIPT
##############################################################################

# Specifications
mlmethod = 'elasticnet5_int0'

# Make summary statistics
make_sumstats(['AT'], ['\\text{Total Assets}_{it}'])

# GMMs to run
run_GMMs_byh([''], False, mlmethod, [''], True, True)

# Decomposition table
make_decomp_table([''], False, mlmethod, [''], True, True)

# PT Estimations
PT_estimation(True, [''], False, mlmethod, [''], True, True)

# X/S test
XS_noiseestimation('EQUITY_VOL', 10, 'Trailing Five-Year Equity Volatility', True, [''], False, mlmethod, [''], True, True)

# Decomposition of CG coefficient
CG_nonoise(True, [''], False, mlmethod, [''], True, True)


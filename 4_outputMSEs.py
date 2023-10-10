import os
import pandas as pd
import numpy as np
from functools import reduce
import random
from sklearn.utils import resample
from tabulate import tabulate

random.seed(33)

###############################################################################
# TO BE SET BY USER

# Working directories
maindir = ''
codedir = maindir + ''
datadir = maindir + ''
outputdir = maindir + ''
figdir = maindir + ''
forecastdir = maindir + ''
###############################################################################

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

# Function to bootstrap single dataset
def bootstrap_single(data, clusterby = None, rs = 33):
    if clusterby == None:
        data_bs = resample(data, replace = True)
    else:
        ids = data[[clusterby]].drop_duplicates()
        data_bs = ids.sample(frac = 1, replace = True, random_state = rs)
        data_bs = data_bs.merge(data, on = clusterby, how = 'left')
    return data_bs.reset_index()

# Function to bootstrap standard error of mean
def bootstrap_semean(dfh, meanvars, Nb = 1000, clusterby = None):
    out = [bootstrap_single(dfh, clusterby, i)[meanvars].mean() for i in range(Nb)]
    # Calculate standard errors
    out = pd.concat(out, axis = 1).std(axis = 1)
    return out

# Function to read and combine MSEs for one type-horizon-frequency
def read_MSE_thf(ftype, estimators, horizon, freq, price_div, DM_cluster = None, winsor = 10,
              normalize_sqeps = True, normalize_variance = False):
    # Create estimator name based on horizon
    if freq == 'a' and horizon >= 3:
        e_labels = [e + '0_int0' for e in estimators]
        rolling = False
    else:
        e_labels = [e + '5_int0' for e in estimators]
        rolling = True
    # Specify which forecast type you're reading in
    if ftype == 'ae':
        base = 'ae_'
    elif ftype == 'af':
        base = 'af_'
    elif ftype == 'e':
        base = ''
    end = '_sic2'
    if price_div:
        end += '_scaled'
    end += '.dta'
    # Import it dataset
    os.chdir(datadir)
    datait = pd.read_sas('datait_' + freq + str(horizon) + '.sas7bdat', format = 'sas7bdat', encoding = 'latin-1')
    datait.columns = map(str.upper, datait.columns)
    datait['GVKEY'] = datait['GVKEY'].astype(int)
    # Import forecasts
    keeps = ['GVKEY', 'PENDS', 'PREDICTED_LEAD_EPS']
    os.chdir(forecastdir)
    fbase = ['forecastdf_' + base + freq + str(horizon) + '_' + e + end for e in e_labels]
    fs = [pd.read_stata(fbase[i])[keeps] for i in range(len(fbase))]
    # Adjust e_labels so they match across horizons and rename forecasts
    if rolling:
        e_labels = [x.replace('5_int0', '') for x in e_labels] 
    else:
        e_labels = [x.replace('0_int0', '') for x in e_labels] 
    fs = [fs[i].rename(columns = {'PREDICTED_LEAD_EPS': e_labels[i] + '_Fm' + ftype}) for i in range(len(fbase))]
    # Merge all statistical forecasts
    forecastdf = reduce(lambda x,y: pd.merge(x, y, on = ['GVKEY', 'PENDS']), fs)
    # Collect analyst forecasts, prices, and current EPS
    os.chdir(datadir)
    adata = pd.read_csv('consendata_' + freq + str(horizon) + '.csv')
    adata['PENDS'] = adata['PENDS'].astype('datetime64[ns]')
    adata = adata[['GVKEY', 'PENDS', 'F_MEAN', 'LEAD_EPS']].merge(datait[['GVKEY', 'PENDS', 'PRCC', 'EPS']], on = ['GVKEY', 'PENDS'])
    # Adjust if you're dividing by price
    if price_div:
        adata[['F_MEAN', 'LEAD_EPS', 'EPS']] = adata[['F_MEAN', 'LEAD_EPS', 'EPS']].div(adata['PRCC'], axis = 0)
    # Merge analyst with statistical forecasts
    keeps_a = ['GVKEY', 'PENDS', 'F_MEAN', 'LEAD_EPS', 'EPS']
    forecastdf['GVKEY'] = forecastdf['GVKEY'].astype(int)
    forecastdf = forecastdf.merge(adata[keeps_a], on = ['GVKEY', 'PENDS']).rename(columns = {'EPS':'randomwalk'})
    # Calculate MSEs for each forecast and winsorize if requested
    fvars = [x for x in forecastdf.columns if '_Fm' in x]
    if ftype == 'e':
        fvars = ['randomwalk'] + fvars
    fvars = ['F_MEAN'] + fvars
    sqerrdf = forecastdf.copy()
    for f in fvars:
        sqerrdf[f] = (sqerrdf[f] - sqerrdf['LEAD_EPS'])**2
    if winsor > 0:
        sqerrdf = clip_outlier_iqr(sqerrdf, fvars, winsor)
    # Calculate MSEs
    MSEs = sqerrdf[fvars].mean()
    # Calculate Diebold-Marino statistics for each forecast compared to analyst forecast
    DMdf = sqerrdf.copy()
    for f in fvars[1:] + [fvars[0]]:
        DMdf[f] = DMdf[f] - DMdf['F_MEAN']
    DM_mean = DMdf[fvars].mean()
    DMdf['YEAR'] = DMdf['PENDS'].dt.year
    bs_DM = bootstrap_semean(DMdf, fvars, Nb = 1000, clusterby = DM_cluster)
    DMs = DM_mean / bs_DM
    # Normalization of MSEs
    if price_div:
        datait['LEAD_EPS'] = datait['LEAD_EPS'] / datait['PRCC']
    if normalize_sqeps:
        norm = (datait['LEAD_EPS']**2).mean()/100
    elif normalize_variance:
        norm = datait['LEAD_EPS'].var()/100
    else:
        norm = 1
    MSEs = MSEs / norm
    return [MSEs, DMs]

# Function to collect MSEs for one type
def tabulate_MSEs_va(ftype, estimators, labels, price_div, DM_cluster = None, winsor = 10,
              normalize_sqeps = True, normalize_variance = False):
    # Collect everything
    n_q = 4
    n_a = 4
    out_q = [read_MSE_thf(ftype, estimators, h, 'q', price_div, DM_cluster, winsor, normalize_sqeps, normalize_variance) for h in range(1,1+n_q)]
    out_a = [read_MSE_thf(ftype, estimators, h, 'a', price_div, DM_cluster, winsor, normalize_sqeps, normalize_variance) for h in range(1,1+n_a)]
    # Make dataframe of results, ordering properly so you can stack DM statistics under MSEs
    df_mse = pd.DataFrame(columns = ['Horizon'] + out_q[0][0].index.to_list())
    df_DM = pd.DataFrame(columns = ['Horizon'] + out_q[0][1].index.to_list())
    j = 0
    for h in range(n_q):
        row = out_q[h][0].copy()
        row['Horizon'] = str(h+1) + ' Quarters'
        row['order'] = j
        df_mse = df_mse.append(row.to_frame().T)
        row = out_q[h][1].copy()
        row['Horizon'] = ''*j
        row['order'] = j
        df_DM = df_DM.append(row.to_frame().T)
        j += 1
    for h in range(n_a):
        row = out_a[h][0].copy()
        row['Horizon'] = str(h+1) + ' Years'
        row['order'] = j
        df_mse = df_mse.append(row.to_frame().T)
        row = out_a[h][1].copy()
        row['Horizon'] = ''*j
        row['order'] = j
        df_DM = df_DM.append(row.to_frame().T)
        j += 1
    # Format to text and add paranthesis around DM statistics
    ncols = [x for x in df_mse.columns if 'Horizon' not in x]
    df_mse[ncols] = df_mse[ncols].applymap(float).round(2).applymap(str) + '%'
    df_mse['order'] = '(' + df_mse['order'].replace('\%', '') + ')'
    df_DM[ncols] = '(' + df_DM[ncols].applymap(float).round(2).applymap(str) + ')'
    df_DM[ncols] = df_DM[ncols].replace('(nan)', '')
    # Merge stacking DM under MSEs
    df_mse['order2'] = 1
    df_DM['order2'] = 2
    df = pd.concat([df_mse, df_DM]).sort_values(['order', 'order2']).drop(columns = ['order', 'order2'])
    # Format dataframe and output to TEX table
    new_cols = ['Horizon', 'Analyst']
    if ftype == 'e':
        new_cols += ['Random Walk']
    new_cols += labels
    df.columns = new_cols
    tex = tabulate(df, showindex = False, tablefmt = 'latex_booktabs', headers = 'keys', stralign = 'center')
    # Some final cleaning
    tex = tex.replace('tabular}{c', 'tabular}{l')
    tex = tex.replace('Horizon', 'Horizon: $h$')
    n_m = len(estimators)
    if ftype == 'e':
        n_m += 1
        lab = '{$MSE_h^e$}'
    else:
        lab = '{$MSE_h^{e+a}$}'
    tex = tex.replace('\\toprule', '\\toprule\n & $MSE^a_h$ & \multicolumn{' + str(n_m) + '}{c}' + lab + '\\\ \n \cmidrule(lr){2-2} \cmidrule{3-' + str(3+n_m-1) + '}')
    # Output
    print(tex)
    os.chdir(outputdir)
    if normalize_variance == True:
        with open('msetable_' + ftype + '_normalizevariance.tex','w') as output:
            output.write(tex)
    elif normalize_sqeps == True:
        with open('msetable_' + ftype + '_normalizesqeps.tex','w') as output:
            output.write(tex)
    

# Make two tables
estimators = ['elasticnet', 'randomforest', 'gradientboostedtree']
labels = ['Elastic Net', 'Random Forest', 'Boosted Trees']
cluster = 'YEAR'
out = tabulate_MSEs_va('e', estimators, labels, True,  DM_cluster = cluster, winsor = 10)
out = tabulate_MSEs_va('ae', estimators, labels, True,  DM_cluster = cluster, winsor = 10)
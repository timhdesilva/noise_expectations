import pandas as pd
import numpy as np
import pickle
import os


###############################################################################
# TO BE SET BY USER

# Working directories
maindir = ''
datadir = maindir + ''
outputdir_f = maindir + ''
outputdir_m = maindir + ''
codedir = maindir + ''
###############################################################################


def output_analystdata(horizon, freq):
    
    # Read in data
    os.chdir(datadir)
    inputfile = 'dataitj_' + freq + str(horizon) + '.sas7bdat'
    outfile_a = 'consendata_' + freq + str(horizon)
    out_mse = 'mse_' + freq + str(horizon)
    out_f = 'forecastdf_' + freq + str(horizon)
    data = pd.read_sas(inputfile, format = 'sas7bdat', encoding = 'latin-1')
    data.columns = map(str.upper, data.columns)
    
    
    
    ###############################################################################
    # CALCULATE CONSENSUS ANALYST FORECAST FOR EACH FIRM-YEAR IN DIFFERENT WAYS
    ###############################################################################
    
    # Median analyst forecast
    data['F_MEDIAN'] = data.groupby(['GVKEY', 'PENDS'])['FCAST'].transform('median')
    
    # Median analyst forecast among firms followed by at least 4 analysts
    data['N_ANALYSTS'] = data.groupby(['GVKEY', 'PENDS'])['FCAST'].transform('count')
    data['F_MEDIAN_GE4'] = data.F_MEDIAN
    data.loc[data.N_ANALYSTS < 4, 'F_MEDIAN_GE4'] = np.nan
    
    # Mean analyst forecast
    data['F_MEAN'] = data.groupby(['GVKEY', 'PENDS'])['FCAST'].transform('mean')
    
    # Weighted-mean analyst forecast, by inverse of realized MSE
    data['FCAST_INVMSE'] = (data.FCAST - data.LEAD_EPS)**(-2)
    fcastweighted = data[['FCAST', 'FCAST_INVMSE']].groupby([data.GVKEY, data.PENDS])\
                .apply(lambda x: np.average(x, weights=x.FCAST_INVMSE, axis = 0))
    fcastweighted = pd.DataFrame(data = fcastweighted.values.tolist(),
                                  index = fcastweighted.index)\
                                  .iloc[:,0].reset_index()
    fcastweighted.rename(columns = {0:'F_WMEAN'}, inplace = True)
    data = pd.merge(data, fcastweighted, how = 'left', on = ['GVKEY', 'PENDS'])
    del fcastweighted
    
    # Best ex-post analyst forecast MSE
    data['MSE_ANALYST'] = (data.FCAST - data.LEAD_EPS)**2
    data['MSE_BEST'] = data.groupby(['GVKEY', 'PENDS'])['MSE_ANALYST'].transform('min')
    bestfcast = data.loc[(data.MSE_BEST == data.MSE_ANALYST), ['GVKEY', 'PENDS', 'FCAST']]\
                .drop_duplicates(['GVKEY', 'PENDS']) # drop duplicates due to symmetry
    bestfcast.rename(columns = {'FCAST':'F_BEST'}, inplace = True)
    data = pd.merge(data, bestfcast, how = 'left', on = ['GVKEY', 'PENDS'])
    del bestfcast
    
    # Export consensus forecast data
    keeps = ['GVKEY', 'PENDS', 'F_MEDIAN', 'F_MEDIAN_GE4', 'F_MEAN', 'F_WMEAN', 'F_BEST', 'LEAD_EPS']
    consendata = data[keeps].drop_duplicates()
    os.chdir(datadir)
    consendata.to_csv(outfile_a + '.csv', index = False)
    del keeps
    
    
    
    ###############################################################################
    # CALCULATE MSE OF CONSENSUS FORECASTS BY YEAR
    ###############################################################################
    
    # Keep only forecasts and year and merge in EPS
    keeps = ['GVKEY', 'PENDS', 'T_DATADATE', 'F_MEDIAN', 'F_MEDIAN_GE4', 'F_MEAN', 'F_WMEAN', 'MSE_BEST', 'LEAD_EPS']
    fmsedata = data[keeps].copy().drop_duplicates()
    
    # Calculate MSE by year for each consensus forecast, replacing missing values
    #   from weighted average with zero because one analyst got things right here.
    fmsedata['MSE_MEDIAN'] = (fmsedata.F_MEDIAN - fmsedata.LEAD_EPS)**2
    fmsedata['MSE_MEDIAN_GE4'] = (fmsedata.F_MEDIAN_GE4 - fmsedata.LEAD_EPS)**2
    fmsedata['MSE_MEAN'] = (fmsedata.F_MEAN - fmsedata.LEAD_EPS)**2
    fmsedata['MSE_WMEAN'] = (fmsedata.F_WMEAN - fmsedata.LEAD_EPS)**2
    fmsedata.MSE_WMEAN.fillna(0, inplace = True)
    
    # Merge in price and calculate price MSEs
    datait = pd.read_sas(inputfile.replace('itj', 'it'), format = 'sas7bdat', encoding = 'latin-1')
    datait.columns = map(str.upper, datait.columns)
    fmsedata = fmsedata.merge(datait[['GVKEY', 'PENDS', 'PRCC']])
    fmsedata['MSE_MEDIAN_PRCC'] = (fmsedata.F_MEDIAN/fmsedata.PRCC - fmsedata.LEAD_EPS/fmsedata.PRCC)**2
    fmsedata['MSE_MEDIAN_GE4_PRCC'] = (fmsedata.F_MEDIAN_GE4/fmsedata.PRCC - fmsedata.LEAD_EPS/fmsedata.PRCC)**2
    fmsedata['MSE_MEAN_PRCC'] = (fmsedata.F_MEAN/fmsedata.PRCC - fmsedata.LEAD_EPS/fmsedata.PRCC)**2
    fmsedata['MSE_WMEAN_PRCC'] = (fmsedata.F_WMEAN/fmsedata.PRCC - fmsedata.LEAD_EPS/fmsedata.PRCC)**2
    fmsedata.MSE_WMEAN_PRCC.fillna(0, inplace = True)
    
    # Calculate MSEs
    MSE_analysts = fmsedata[[x for x in fmsedata.columns if 'MSE' in x]].groupby(fmsedata.T_DATADATE).mean()
    
    # Calculate median analyst MSE, instead of MSE of median analyst
    median_MSE = data[['GVKEY', 'PENDS', 'T_DATADATE', 'MSE_ANALYST']].copy()
    median_MSE['MEDIAN_MSE'] = median_MSE.groupby(['GVKEY', 'PENDS'])['MSE_ANALYST'].transform('median')
    median_MSE = median_MSE.groupby('T_DATADATE')['MEDIAN_MSE'].mean()
    
    # Output time-series of analyst MSE and dataframe of forecasts
    MSE_analysts = MSE_analysts.merge(median_MSE, left_index = True, right_index = True)
    os.chdir(outputdir_m)
    with open(out_mse + '_analysts.pcl', 'wb') as f:
        pickle.dump(MSE_analysts, f)
    f.close()
    os.chdir(outputdir_f)
    with open(out_f + '_analysts.pcl', 'wb') as f:
        pickle.dump(fmsedata, f)
    f.close()

for h in range(1,6):
    for f in ['a', 'q']:
        if f == 'a' or h <= 4:
            output_analystdata(h, f)
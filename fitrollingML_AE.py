import pandas as pd
import os
import sys 
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
import multiprocessing
import warnings



###############################################################################
# TO BE SET BY USER

# Working directories
maindir = ''
datadir = maindir + ''
outputdir_f = maindir + ''
outputdir_m = maindir + ''
codedir = maindir + ''
###############################################################################

##############################################################################
# GLOBAL FILE PARAMETERS
consensus = 'F_MEAN'
include_2lags = True
##############################################################################


##############################################################################
# FUNCTION DEFINITION FOR RUNNING ML ONLY
##############################################################################
# Define function to output MSEs depending on inputs. This function
#   does not return anything - it outputs a dataframe in a pickle containing
#   the MSEs from the ML
def output_MSE(freq, horizon, method, price_div, roll_yrs, int_order = 0, sicdig = 2):
#    freq: frequency of forecasting
#    horizon: horizon for forecasting
#    method = method to use - see mlwrapper.py
#    price_div: = True if you want to divide by price
#    roll_yrs: number of years of past data you want to use
#    int_order: choose highest level of interactions/polynomials (if no, no interactions)
#    sicdig: number of SIC code digits to use for industry fixed effects (1 or 2)


    # Print parameters
    print('-------------------------------------------------------------------------------------------------------------------------\n',\
            'Chosen parameters:\n', 'method = ', method, ', Interaction/Polynomial Power = ', int_order,\
            ', SIC digits = ', sicdig, ', Horizon = ', horizon, ', Frequency = ', freq, '\nDivided EPS by Price = ', price_div, \
            ', Number of CPUs available:', multiprocessing.cpu_count(), ', Rolling years:', roll_yrs, \
          '\n-----------------------------------------------------------------------------------------------------------------------\n')
    
    # Import functions from directory
    sys.path.append(codedir)
    import fxns_msefitting as tds
    
    # Read in data
    os.chdir(datadir)
    inputfile = 'datait_' + freq + str(horizon) + '.sas7bdat'
    inputconsen = 'consendata_'+ freq + str(horizon) + '.csv'
    data = pd.read_sas(inputfile, format = 'sas7bdat', encoding = 'latin-1')
    data.columns = map(str.upper, data.columns)
        
    # Choose SIC digit
    if sicdig == 2:
        data = data.rename(columns = {'SIC_2':'SIC'})
        del data['SIC_1']
    else:
        data = data.rename(columns = {'SIC_1':'SIC'})
        del data['SIC_2']

    # Read in consensus analyst data and merge
    os.chdir(datadir)
    consendata = pd.read_csv(inputconsen, dtype = 'str')
    consendata['PENDS'] = pd.to_datetime(consendata.PENDS, format = '%Y-%m-%d')
    numeric_cols = list(consendata.columns[2:])
    consendata[numeric_cols] = consendata[numeric_cols].apply(pd.to_numeric)
    consendata['F_ANALYST'] = consendata[consensus]
    data = data.merge(consendata[['GVKEY', 'PENDS', 'F_ANALYST']], how = 'inner')
    
    # Divide by price
    if price_div:
        eps_vars = ['F_ANALYST', 'LEAD_EPS', 'EPS']
        if include_2lags:
            eps_vars += ['LAG_EPS', 'LAG2_EPS']
        data[eps_vars] = data[eps_vars].div(data['PRCC'], axis = 0)
 
    # Drop columns not needed for ML and rearrange columns in the order needed 
    # for running MLs with the dependent variable first followed by all X's.
    idcols = ['PENDS', 'ANNDATS_ACTUAL', 'LEAD_PENDS', 'LEAD_ANNDATS_ACTUAL', 'PERMNO', 
             'DATADATE', 'GVKEY', 'TICKER', 'QTR', 'LEAD_EPS', 'T_DATADATE']
    cols = [x for x in data.columns if x not in idcols]
    cols.insert(0, 'T_DATADATE')
    cols.insert(1, 'LEAD_EPS')
    mldata = data[cols].copy().sort_values(['T_DATADATE'])
    
    # Fill missing values of contemporaneous variables using industry-time means and pull back for lags
    drops = ['LEAD_EPS', 'T_DATADATE', 'PRCC', 'N_ANALYSTS', 'SIC', 'F_ANALYST']
    contemp_cols = [x for x in mldata.columns if 'LAG' not in x and x not in drops]
    for col in contemp_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mldata[col] = mldata.groupby(['SIC', 'T_DATADATE'])[col].apply(lambda x: x.fillna(x.median()))
        mldata[col] = mldata.groupby(['T_DATADATE'])[col].apply(lambda x: x.fillna(x.median())) # replace any missing values with time means
        mldata[col] = mldata[col].fillna(mldata[col].median()) # fill any remaining missings with overall median
        mldata[col] = mldata[col].fillna(0) # fill any remaining missings with zero
        # use medians to fill any missings
        if include_2lags:
            if col not in ['RET_1', 'RET_12_2', 'EQUITY_VOL']: # these are the variables that don't lag
                lag1_col = 'LAG_' + col
                lag2_col = 'LAG2_' + col
                mldata[lag1_col] = mldata[lag1_col].fillna(mldata[col])
                mldata[lag2_col] = mldata[lag2_col].fillna(mldata[lag1_col])
    
    # Create dummy variables industry codes  
    mldata = pd.get_dummies(mldata, columns = ['SIC'], drop_first = True)
    
    # Print variables used in forecasting
    print('Features used for forecasting (excluding interactions):\n', [c for c in mldata.columns if c != 'LEAD_EPS' and c != 'T_DATADATE'])
    
    # Create polynomials and/or interactions (without intercept b/c will demean) using only X's - this requires creating a new dataframe
    if int_order >= 2:
        extraterms = PolynomialFeatures(int_order, interaction_only = False, include_bias = False)
        colnames = extraterms.fit(mldata.iloc[:,2:]).get_feature_names(mldata.iloc[:,2:].columns)
        mldata_int = pd.DataFrame(extraterms.fit_transform(mldata.iloc[:,2:]), columns = colnames, index = mldata.index)
        mldata_int.insert(0, 'T_DATADATE', mldata.T_DATADATE.values)
        mldata_int.insert(1, 'LEAD_EPS', mldata.LEAD_EPS.values)
    else:
        mldata_int = mldata
    
    # Run ML algorithm of choice
    starttime = datetime.now()
    [mse, forecast] = tds.calc_rollingMSE(mldata_int, method, roll_yrs, parallel = True)
    time = datetime.now()-starttime
    print('Model fitting time:', time)

    # Collect information for forecasts by merging via INDEX
    forecast = forecast.merge(data[idcols[:-2]], left_index = True, right_index = True)
    forecast.sort_values(by = ['GVKEY', 'PENDS'], inplace = True)
    
    # Prepare for output
    method_out = method.replace(' ', '').lower()
    outfile_m = 'mse_ae_' + freq + str(horizon) + '_'
    outfile_f = 'forecastdf_ae_' + freq + str(horizon) + '_'
    filename = outfile_m + method_out + str(roll_yrs) + '_int' + str(int_order)  + '_sic' + str(sicdig)
    filename_f = outfile_f + method_out + str(roll_yrs) + '_int' + str(int_order)  + '_sic' + str(sicdig)
    if price_div:
        filename += '_scaled'
        filename_f += '_scaled'
    filename += '.csv'
    filename_f += '.dta'
    
    # Output
    os.chdir(outputdir_m)
    mse.to_csv(filename)
    os.chdir(outputdir_f)
    forecast.to_stata(filename_f)
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:55:08 2019

@author: javie
"""

import pandas as pd
import numpy as np
import datetime
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from imputation import _impute_by_band, _impute_with_pop_means

modeling_data_output = 'C:/Users/Owner/Desktop/2019/Girthier Model/Modeling Datasets/'

# Settings for data building stage
source_path = 'C:/Users/Owner/Desktop/2019/Girthier Model/Source Data/'

#salary_data_path = 'C:/Users/javie/OneDrive/Documentos/Fantasy Football/2018/Salary/'

today_date = datetime.date.today()
prep_data_output_suffix = 'OrigGirth_minus7day_MonsterCocks'
#prep_data_output_suffix = today_date.strftime('%Y%m%d')

def _1_upload_csv():
    print('\n', '****** STEP 1: DATA UPLOAD AND EXPLORATION ******')
    df_player_data = pd.read_csv(source_path + 'OrigGirth_minus7day_MonsterCocks.csv')
    df_player_data.rename(columns={'Unnamed: 0' : 'row_num'}, inplace=True)
       
    print('\n', 'Data set shape:', df_player_data.shape)
    print('\nDuplicated PlayerLookup:')
    print(df_player_data[df_player_data.duplicated(subset='PlayerLookup', keep=False)][['row_num', 'PlayerLookup']]) #Check syntax...
    #df_player_data = df_player_data[~df_player_data['row_num'].isin([2084, 2251])]
    
    df_player_data.dropna(subset = ['PlayerLookup'], inplace = True)
    # Dropping Duplicates
    df_player_data.drop_duplicates(subset = 'PlayerLookup', keep = 'first', inplace = True)
    
    # Cleaning and renaming columns
    df_player_data.set_index('PlayerLookup', inplace=True)
    df_player_data.drop(['row_num'], axis=1, inplace=True)
    df_player_data.rename(columns = {'Obj: FanstasyPoints' : 'Obj:_FDPoints'
                                     , 'Home?' : 'home_flag'}, inplace = True)
    
    # Recasting data types
    df_player_data['FD_Salary'] = pd.to_numeric(df_player_data['FD_Salary'], errors='coerce')
    #df_player_data['bmon_mC'] = pd.to_numeric(df_player_data['bmon_mC'], errors='coerce')

    # Creating Home Flag
    #df_player_data['home_flag'] = np.where(df_player_data['Opp'].str.find('@') == -1, 1 , 0)
    
    # Fixing and Encoding Positions
    df_player_data['Position'] = df_player_data['Position'].str.split('/', expand=True)[0]
    df_player_data['pos_SG'] = np.where(df_player_data['Position'] == 'SG', 1, 0)
    df_player_data['pos_PF'] = np.where(df_player_data['Position'] == 'PF', 1, 0)
    df_player_data['pos_PG'] = np.where(df_player_data['Position'] == 'PG', 1, 0)
    df_player_data['pos_SF'] = np.where(df_player_data['Position'] == 'SF', 1, 0)
    df_player_data['pos_C'] = np.where(df_player_data['Position'] == 'C', 1, 0)
    
    # Fixing and Encoding BMon Positions
    df_player_data['EPos_SG'] = np.where(df_player_data['Epos'] == 'SG', 1, 0)
    df_player_data['EPos_PF'] = np.where(df_player_data['Epos'] == 'PF', 1, 0)
    df_player_data['EPos_PG'] = np.where(df_player_data['Epos'] == 'PG', 1, 0)
    df_player_data['EPos_SF'] = np.where(df_player_data['Epos'] == 'SF', 1, 0)
    df_player_data['EPos_C'] = np.where(df_player_data['Epos'] == 'C', 1, 0)
    
    # Encoding Douches' Opinions
    df_player_data['douche_G_both'] = np.where(df_player_data['DouchePickG'] == 'BOTH', 1, 0)
    df_player_data['douche_G_cash'] = np.where(df_player_data['DouchePickG'] == 'CASH', 1, 0)
    df_player_data['douche_G_gpp'] = np.where(df_player_data['DouchePickG'] == 'GPP', 1, 0)
    
    df_player_data['douche_J_both'] = np.where(df_player_data['Douches_pickJ'] == 'BOTH', 1, 0)
    df_player_data['douche_J_cash'] = np.where(df_player_data['Douches_pickJ'] == 'CASH', 1, 0)
    df_player_data['douche_J_gpp'] = np.where(df_player_data['Douches_pickJ'] == 'GPP', 1, 0)
    
    # Encoding Likes of Fellow Fags
    df_player_data['likes_missing'] = np.where(pd.isnull(df_player_data['Likes_of_Fellow_Fags']), 1, 0)
    df_player_data['likes_0'] = np.where(df_player_data['Likes_of_Fellow_Fags'] == 0, 1, 0)
    df_player_data['likes_1_50'] = np.where((df_player_data['Likes_of_Fellow_Fags'] > 0) & 
                  (df_player_data['Likes_of_Fellow_Fags'] <= 50), 1, 0)
    df_player_data['likes_51_75'] = np.where((df_player_data['Likes_of_Fellow_Fags'] > 50) & 
                  (df_player_data['Likes_of_Fellow_Fags'] <= 75), 1, 0)
    df_player_data['likes_76_99'] = np.where((df_player_data['Likes_of_Fellow_Fags'] > 75) & 
                  (df_player_data['Likes_of_Fellow_Fags'] <= 99), 1, 0)
    df_player_data['likes_100'] = np.where(df_player_data['Likes_of_Fellow_Fags'] == 100, 1, 0)
    
    # Processing Odds field
    df_player_data[['odds_1', 'odds_2']] = df_player_data['Odds'].str.split(n=3, expand=True)[[1,2]]
    df_player_data['odds_1'] = df_player_data['odds_1'].astype(float)
    df_player_data['odds_2'] = df_player_data['odds_2'].astype(float)
    df_player_data.drop('Odds', axis=1, inplace=True)
     
    excel_writer = pd.ExcelWriter('1_EDA_' + str(prep_data_output_suffix) + '.xlsx')
        
    #Exporting Data Types
    data_types = df_player_data.dtypes.to_frame(name='data_type')
    data_types.index = data_types.index.astype(str)
    data_types['data_type'] = data_types['data_type'].astype(str)
        
    print('\n', 'Exporting Data Type Summary to Excel Output...')
    data_types.to_excel(excel_writer, sheet_name='Data Types')
        
    print('\n', 'New Shape:', df_player_data.shape)
        
    #Dependent Variable Histogram
    df_player_data['Obj:_FDPoints'].hist()
    
    #Descriptive Statistics
    missing_counts = df_player_data.isnull().sum()
    missing_counts_df = missing_counts.to_frame(name='num_missing')
    missing_counts_df.index = missing_counts_df.index.astype(str)

    print('\n', 'Exporting Missing Value Counts to Excel Output...')
    missing_counts_df.to_excel(excel_writer, sheet_name='Missing Value Counts')     
    missing_counts_gt_0 = missing_counts[missing_counts > 0]
        
    print('\n', '# of columns with at least 1 missing field:', 
          len(missing_counts_gt_0))
        
    descriptive_stats = df_player_data.describe(percentiles=[.01, 0.05, 0.25, 0.5, 
                                                      .75, .95, .99]).transpose()
    descriptive_stats['pct_missing'] = 1 - (descriptive_stats['count'] / len(df_player_data))
    
    descriptive_stats.sort_values(['pct_missing'], ascending=False, inplace=True)
    print('\n', 'Exporting Descriptive Stats (Numeric Fields) to Excel Output...')
    descriptive_stats.to_excel(excel_writer, sheet_name='Descriptive Stats - Num')
        
    descriptive_stats = df_player_data.describe(include=[np.object]).transpose()
    print('\n', 'Exporting Descriptive Stats (String Fields) to Excel Output...')
    descriptive_stats.to_excel(excel_writer, sheet_name='Descriptive Stats - String')
    
    # Correlations against target variable
    numeric_fields = df_player_data.select_dtypes(include=['number'])
    corr_vs_FDPoints = np.zeros((len(numeric_fields.columns), 3), dtype=object)
    
    for i, col in enumerate(numeric_fields.columns[numeric_fields.columns != 'Obj:_FDPoints']):
        df_clean = numeric_fields[['Obj:_FDPoints', col]].dropna()
        corr_tuple = pearsonr(df_clean['Obj:_FDPoints'], df_clean[col])
        
        corr_vs_FDPoints[i, 0] = col
        corr_vs_FDPoints[i, 1] = corr_tuple[0]
        corr_vs_FDPoints[i, 2] = corr_tuple[1]        
        
    print('\n', 'Exporting Correlations vs. Target to Excel Output...')
    corr_vs_FDPoints_df = pd.DataFrame(corr_vs_FDPoints, columns=['variable', 'corr', 'p-value'])
    corr_vs_FDPoints_df.set_index('variable', inplace=True)
    corr_vs_FDPoints_df.sort_values(['p-value'], inplace=True)
    corr_vs_FDPoints_df.to_excel(excel_writer, sheet_name='Corrs Against Target')
    
    #Time Series
    means_by_date = df_player_data.groupby('Date').mean()
    print('\n', '# of Dates: ', len(means_by_date))
    print('\n', 'Exporting Weekly Averages to Excel Output...')  
    means_by_date.to_excel(excel_writer, sheet_name='Weekly Avgs')
    
    #Correlations Against Linear Trend    
    corr_vs_trend = np.zeros((len(means_by_date.columns), 3), dtype=object)
    means_by_date['linear'] = np.arange(0, len(means_by_date))

    for i, col in enumerate(means_by_date.columns[means_by_date.columns != 'linear']):
        df_clean = means_by_date[['linear', col]].dropna()
        corr_tuple = pearsonr(df_clean['linear'], df_clean[col])
        
        corr_vs_trend[i, 0] = col
        corr_vs_trend[i, 1] = corr_tuple[0]
        corr_vs_trend[i, 2] = corr_tuple[1]
    
    print('\n', 'Exporting Correlations Weekly Avg. Vs. Linear Trend to Excel Output...')
    corr_vs_trend_df = pd.DataFrame(corr_vs_trend, columns=['variable', 'corr', 'p-value'])
    corr_vs_trend_df.set_index('variable', inplace=True)
    corr_vs_trend_df.sort_values(['p-value'], inplace=True)
    corr_vs_trend_df.to_excel(excel_writer, sheet_name='Corrs Against Trend')

    excel_writer.save()
    return df_player_data;

def salary_bands(df_player_data):
    if df_player_data['FD_Salary'] > 0 and df_player_data['FD_Salary'] <= 3600:
        return 1
    if df_player_data['FD_Salary'] <= 4400:
        return 2
    if df_player_data['FD_Salary'] <= 6100:
        return 3
    if df_player_data['FD_Salary'] <= 7900:
        return 4
    if df_player_data['FD_Salary'] > 7900:
        return 5
    return ;

def _2_create_salary_bands(df_player_data):
    print('\n', '****** STEP 2: CREATING SALARY BANDS ******')
    excel_writer = pd.ExcelWriter('2_salary_bands_' + str(prep_data_output_suffix) + '.xlsx')  
    
    df_player_data['salary_band'] = df_player_data.apply(salary_bands, axis=1)
    
    df_salary_band_stats = df_player_data.groupby('salary_band')['FD_Salary'].agg(['count', 'mean', 'min', 'max'])
    print('Salary band stats:')
    print(df_salary_band_stats)
    df_salary_band_stats.to_excel(excel_writer, sheet_name='Bands')

    return df_player_data;

def _3_split_datasets(df_player_data):
    
    print('\n', '****** STEP 3: TRAIN-TEST SPLIT ******')
    # Creating training and test datasets
    
    X = df_player_data.copy()
    X.drop(['Obj:_FDPoints'], axis=1, inplace=True)
    y = df_player_data['Obj:_FDPoints']
    
#    X = df_player_data.iloc[:, 1:]
#    y = df_player_data.iloc[:, 0]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
        
    print('\n', '**** Training Data ****' )
    print('\n', '\tX: ', X_train.shape)
    print('\n', '\ty: ', y_train.shape)
    print()
        
    print('\n', '**** Test Data ****')
    print('\n', '\tX: ', X_test.shape)
    print('\n', '\ty: ', y_test.shape)
    
    return  X_train, X_test, y_train, y_test;   

def _4_create_imputation_file(X_train):
    print('\n', '****** STEP 4: CREATING IMPUTATION FILE (Training Data Only) ******')
    excel_writer = pd.ExcelWriter('4_df_imputation_values_' + str(prep_data_output_suffix) + '.xlsx')     
    
    # Means by salary band
    df_imputation_values = X_train.groupby('salary_band').mean()
    df_imputation_values.reset_index(inplace=True)
    
    df_imputation_values.drop(['FD_Salary'], axis=1, inplace=True)
    
    df_imputation_values_1 = pd.melt(df_imputation_values, id_vars=['salary_band'],
                                     var_name = 'metric', value_name = 'value')
    
    df_imputation_values_1.to_excel(excel_writer, sheet_name = 'Means by Salary Band')
    
    #Population means (for cases where salary band is missing)
    # Note that this imputes Salary
    df_imputation_values_pop = X_train.mean()
    df_imputation_values_pop.drop(['salary_band'], inplace=True)
    
    df_imputation_values_pop.to_excel(excel_writer, sheet_name = 'Population Means')
    
    # Paste this output in template to create imputation code
    excel_writer.save()
    return ;

def _5_impute_missing_num(X_train, X_test):
    
    print('\n', '****** STEP 5: IMPUTATION (NUMERIC VARS) ******')
    excel_writer = pd.ExcelWriter('5_imputation_num_' + str(prep_data_output_suffix) + '.xlsx')    

    # Imputing Missing Values and Standardizing
    X_train_numeric = X_train.select_dtypes(include=['number'])
    X_test_numeric = X_test.select_dtypes(include=['number'])

    print('\n', 'Exporting Missing Pcts, Mean and Median of Numeric Vars Before Imputation...')  
    X_train_numeric_stats = X_train_numeric.describe().loc[['count', 'mean', '50%']].transpose()
    X_train_numeric_stats['pct_missing'] = 1 - (X_train_numeric_stats['count'] / len(X_train_numeric))
    
    X_test_numeric_stats = X_test_numeric.describe().loc[['count', 'mean', '50%']].transpose()
    X_test_numeric_stats['pct_missing'] = 1 - (X_test_numeric_stats['count'] / len(X_test_numeric))
    
    X_train_numeric_stats.to_excel(excel_writer, sheet_name='X Train (Num) - Before Imp')
    X_test_numeric_stats.to_excel(excel_writer, sheet_name='X Test (Num) - Before Imp')

    X_train_numeric_1 = _impute_by_band(X_train_numeric)    
    X_test_numeric_1 = _impute_by_band(X_test_numeric)   
    
    X_train_numeric_1 = _impute_with_pop_means(X_train_numeric_1)
    X_test_numeric_1 = _impute_with_pop_means(X_test_numeric_1)

    print('\n', 'Exporting Missing Pcts, Mean and Median of Numeric Vars After Imputation...')  
    X_train_numeric_1_stats = X_train_numeric_1.describe().loc[['count', 'mean', '50%']].transpose()
    X_train_numeric_1_stats['pct_missing'] = 1 - (X_train_numeric_1_stats['count'] / len(X_train_numeric))
    
    X_test_numeric_1_stats = X_test_numeric_1.describe().loc[['count', 'mean', '50%']].transpose()
    X_test_numeric_1_stats['pct_missing'] = 1 - (X_test_numeric_1_stats['count'] / len(X_test_numeric))
    
    X_train_numeric_1_stats.to_excel(excel_writer, sheet_name='X Train (Num) - After Imp')
    X_test_numeric_1_stats.to_excel(excel_writer, sheet_name='X Test (Num) - After Imp')

    excel_writer.save()
    return  X_train_numeric_1, X_test_numeric_1;

def _6_impute_missing_text(X_train, X_test, y_train, y_test):
    return ;

def _7_save_modeling_data(X_train, X_test, y_train, y_test):
    
    print('\n', '****** STEP 7: SAVING MODELING DATA ******')
    
    X_train.to_pickle(modeling_data_output + 'X_train_df_' + str(prep_data_output_suffix) + '.pickle')
    X_test.to_pickle(modeling_data_output + 'X_test_df_' + str(prep_data_output_suffix) + '.pickle')
    y_train.to_pickle(modeling_data_output + 'y_train_df_' + str(prep_data_output_suffix) + '.pickle')
    y_test.to_pickle(modeling_data_output + 'y_test_df_' + str(prep_data_output_suffix) + '.pickle')
    
    return ;

def _main_data_prep():
    print('***** DATA PREPARATION *****')
    print('Process started: ',datetime.datetime.now(), '\n')
    #1 - Data import and initial exploration
    df_player_data = _1_upload_csv()

    # Additional cleanup after exploring data
    df_player_data.drop([
                  
                  # 80pct+ missing:
                  'Likes_of_Fellow_Fags'

             
                  ], axis=1, inplace=True)
    print('\n', 'Shape after cleaning up additional fields: ', df_player_data.shape)
    
    
    #2 = Create salary bands
    df_player_data = _2_create_salary_bands(df_player_data)
    
       
    #3 - Train vs. Test split
    X_train, X_test, y_train, y_test = _3_split_datasets(df_player_data)
    
    #4 - Creates master imputation file (avgs. by band) and exports to Excel
    _4_create_imputation_file(X_train)
    #Use output and code template to create imputation code
    
    print('Imputation started: ',datetime.datetime.now(), '\n')
    #5 - Imputing numeric variables
    X_train_numeric_1, X_test_numeric_1 = _5_impute_missing_num(X_train, X_test)
    
    #6 - Imputing categorical variables (Coming sooon...)
    
    #7 - Saving modeling data to file
    _7_save_modeling_data(X_train_numeric_1, X_test_numeric_1, y_train, y_test)
        
    print('Process ended: ',datetime.datetime.now(), '\n')
    return;
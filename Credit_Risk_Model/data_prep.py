###########################################################
#This file cleans Consumer Loan data downloaded from Kaggle
# https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
# It process continuous and descete dependent variables
# Then create dummy variables for the PD model
##########################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Import data
data_file = input("Enter datafile name: ")
loan_data = pd.read_csv(data_file)

'''
prerocess CONTINEOUS variable
Remove the unnecessary test from each row and then turn the remains into a nummerical value

4 variables are processed:
    (1) term
    (2) emp_length: employment length
    (3) issue_d: issue date 
    (4) earliest_cr_line: Earlist Credit Date
    (5) last_pymnt_d: last payment date
    (6) last_credit_pull_d: last credit pull date
    
when process variables 3-6 we assume current date to be 2017-12-01 as the file is a bit outdated
For variables with NaT/NaN values (less than 0.01% of total data), we fill them with earliest available date
'''

#term
#We have two unique values '36months' and '60months'
loan_data['term_int'] = loan_data['term'].map({' 36 months':'36',' 60 months':'60'})
loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])

#emp_length::employment length
loan_data['emp_length_int'] = loan_data['emp_length'].map({'< 1 year':0,
                                                           '1 year':1,
                                                           '2 years':2,
                                                           '3 years':3,
                                                           '4 years':4,
                                                           '5 years':5,
                                                           '6 years':6,
                                                           '7 years':7,
                                                           '8 years':8,
                                                           '9 years':9,
                                                           '10+ years':10,
                                                                 })
loan_data['emp_length_int'] = loan_data['emp_length_int'].fillna(0)

#issue_d::issue date
loan_data['issue_d_string'] = pd.to_datetime(loan_data['issue_d'],format = '%b-%y')
loan_data['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01')-loan_data['issue_d_string'])/np.timedelta64(1,'M')))

'''
earliest_cr_line:: earliest credit date
We worked on data before 1970-01-01 to ensure no negative days.
'''
loan_data['earliest_cr_line_string'] = pd.to_datetime(loan_data['earliest_cr_line'],format = '%b-%y')
loan_data['earliest_cr_line_string']  = loan_data['earliest_cr_line_string'].fillna(pd.to_datetime('1969-01-01'))
loan_data['earliest_cr_line_date_day'] = loan_data['earliest_cr_line_string'].apply(lambda x: (pd.to_datetime('2117-12-01')-x) if x > pd.to_datetime('2020-01-01')
                                         else (pd.to_datetime('2017-12-01')-x))
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric(loan_data['earliest_cr_line_date_day']/np.timedelta64(1,'M')))

#last_pymnt_d::last payment date
loan_data['last_pymnt_d_string'] = pd.to_datetime(loan_data['last_pymnt_d'],format = '%b-%y')
loan_data['last_pymnt_d_string'].fillna(pd.to_datetime('2007-12-01'),inplace = True)
loan_data['mths_since_last_delinq'] = round(pd.to_numeric((pd.to_datetime('2017-12-01')-loan_data['last_pymnt_d_string'])/np.timedelta64(1,'M')))

#last_credit_pull_d:: last credit pull date
loan_data['last_credit_pull_d_string'] = pd.to_datetime(loan_data['last_credit_pull_d'],format = '%b-%y')
loan_data['last_credit_pull_d_string'].fillna(pd.to_datetime('2007-05-01'),inplace = True)
loan_data['mths_since_last_record'] = round(pd.to_numeric((pd.to_datetime('2017-12-01')-loan_data['last_credit_pull_d_string'])/np.timedelta64(1,'M')))

'''
Process 8 DISCRETE Variables
grade, sub_grade, home_ownership, verification_status,loan_status, initial_list_status, purpose, add_state
sub-grade do not give as much info as grade, therefore we ignore it in the first model
We create DUMMY VARIABLES for them
'''
loan_data_dummies = [pd.get_dummies(loan_data['grade'],prefix = 'grade', prefix_sep = ':'),
                    # pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':'),
                     pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':'),
                    pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':')]
    
loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)
loan_data = pd.concat([loan_data,loan_data_dummies],axis = 1)


'''
fill missing values
total_rev_hi_lim: using funded amount 
annual_inc: only 4 missing, use average or alternatively lowest possible
delinq_2yrs, inq_last_6mths, open_acc,total_acc,acc_now_delinq: fill with 0
'''
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace = True)
loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(),inplace = True)
loan_data['delinq_2yrs'].fillna(0.0, inplace = True)
loan_data['inq_last_6mths'].fillna(0, inplace = True)
loan_data['open_acc'].fillna(0, inplace = True)
loan_data['total_acc'].fillna(0, inplace = True)
loan_data['acc_now_delinq'].fillna(0.0, inplace = True)


'''
PD Model Data Preparation
Dependent Variable: loan_status (Default, non-default)
Independent variables: dummy variables, split according to weight of evidence
                       WoE = ln(%good/%bad) 
'''
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off',
                                                                'Late (31-120 days)',
                                                                'Does not meet the credit policy. Status:Charged Off',
                                                                'Default']),0,1)

'''
Train test split 
we set the sice of train dataset to be 80% and test dataset to be 20% 
A specific random state 42 is sellected
This would allow us to perform the exact sam split multiple times and assign the same observations
We first process training set and then the test set
'''
loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis = 1), loan_data['good_bad'], test_size = 0.2, random_state = 42)

for counts in range(0,2):
        
    df_inputs_prepr = loan_data_inputs_train
    df_targets_prepr = loan_data_targets_train
    
    if counts == 1:
        df_inputs_prepr = loan_data_inputs_test
        df_targets_prepr = loan_data_targets_test
    else:
        pass
    '''
    Independent variables - Discrete variables
    grade, sub_grade, home_ownership, verification_status,loan_status, initial_list_status, purpose, add_state
    We observed trends in WoE and decided on how we combine discrete values 
    '''
    #grade - we keep everything
    
    #home_ownership - combine RENT/OTHER/NONE/ANY as they have similar WoE
    df_inputs_prepr['home_ownership: RENT_OTHER_NONE_ANY'] = 1 - df_inputs_prepr['home_ownership:OWN'] - df_inputs_prepr['home_ownership:MORTGAGE']
    
    #check if ND is included in addr_states
    if ['addr_state:ND'] in df_inputs_prepr.columns.values:
        pass
    else:
        df_inputs_prepr['addr_state:ND'] = 0
    
    #addr_states: group those with similar weight of eividence  
    df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'], df_inputs_prepr['addr_state:NE'],
                                                              df_inputs_prepr['addr_state:IA'], df_inputs_prepr['addr_state:NV'],
                                                              df_inputs_prepr['addr_state:FL'], df_inputs_prepr['addr_state:HI'],
                                                              df_inputs_prepr['addr_state:AL']])
    df_inputs_prepr['addr_state:NM_VA'] = sum([df_inputs_prepr['addr_state:NM'], df_inputs_prepr['addr_state:VA']])
    df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'], df_inputs_prepr['addr_state:TN'],
                                                           df_inputs_prepr['addr_state:MO'], df_inputs_prepr['addr_state:LA'],
                                                           df_inputs_prepr['addr_state:MD'], df_inputs_prepr['addr_state:NC']])
    df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_prepr['addr_state:UT'], df_inputs_prepr['addr_state:KY'],
                                                     df_inputs_prepr['addr_state:AZ'], df_inputs_prepr['addr_state:NJ']])
    df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'], df_inputs_prepr['addr_state:MI'],
                                                        df_inputs_prepr['addr_state:PA'], df_inputs_prepr['addr_state:OH'],
                                                        df_inputs_prepr['addr_state:MN']])
    df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'], df_inputs_prepr['addr_state:MA'],
                                                        df_inputs_prepr['addr_state:DE'], df_inputs_prepr['addr_state:SD'],
                                                        df_inputs_prepr['addr_state:IN']])
    df_inputs_prepr['addr_state:GA_WA_OR'] = sum([df_inputs_prepr['addr_state:GA'], df_inputs_prepr['addr_state:WA'],
                                                  df_inputs_prepr['addr_state:OR']])
    df_inputs_prepr['addr_state:WI_MT'] = sum([df_inputs_prepr['addr_state:WI'], df_inputs_prepr['addr_state:MT']])
    df_inputs_prepr['addr_state:IL_CT'] = sum([df_inputs_prepr['addr_state:IL'], df_inputs_prepr['addr_state:CT']])
    df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'], df_inputs_prepr['addr_state:SC'],
                                                           df_inputs_prepr['addr_state:CO'], df_inputs_prepr['addr_state:VT'],
                                                           df_inputs_prepr['addr_state:AK'], df_inputs_prepr['addr_state:MS']])
    df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'], df_inputs_prepr['addr_state:NH'],
                                                           df_inputs_prepr['addr_state:WY'], df_inputs_prepr['addr_state:DC'],
                                                           df_inputs_prepr['addr_state:ME'], df_inputs_prepr['addr_state:ID']])
    
    #verification status- keep as they are as they have different WoE
    
    #purpose
    df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'], df_inputs_prepr['purpose:small_business'],
                                                                           df_inputs_prepr['purpose:wedding'], df_inputs_prepr['purpose:renewable_energy'],
                                                                           df_inputs_prepr['purpose:moving'], df_inputs_prepr['purpose:house']])
    df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'], df_inputs_prepr['purpose:medical'],
                                                         df_inputs_prepr['purpose:vacation']])
    df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'], df_inputs_prepr['purpose:car'],
                                                                  df_inputs_prepr['purpose:home_improvement']])
         
    #Independent variables - Continuous 
    #term
    
    #term:: 60 being the reference category
    df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
    df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)
    
    #emp_length:: 0 being the reference category
    df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
    df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
    df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
    df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
    df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
    df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)
    
    #mths_since_issue_d
    df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:42-48'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:49-52'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:53-64'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:65-84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(85, int(df_inputs_prepr['mths_since_issue_d'].max()))), 1, 0)
    
    #int_rate
    df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548), 1, 0)
    df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
    df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74), 1, 0)
    df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
    df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281), 1, 0)
    
    #mths_since_erliest_cr_line
    df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line']<140),1,0)
    df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line']>=141)&(df_inputs_prepr['mths_since_earliest_cr_line']<=164),1,0)
    df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line']>=165)&(df_inputs_prepr['mths_since_earliest_cr_line']<=247),1,0)
    df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line']>=248)&(df_inputs_prepr['mths_since_earliest_cr_line']<=270),1,0)
    df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line']>=271)&(df_inputs_prepr['mths_since_earliest_cr_line']<=352),1,0)
    df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line']>352),1,0)
    
    #delinq_2yrs
    df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs']==0),1,0)
    df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs']>=1)&(df_inputs_prepr['delinq_2yrs']<=3),1,0)
    df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs']>=4),1,0)
    
    #inq_last_6mths
    df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths']==0),1,0)
    df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths']>=1)&(df_inputs_prepr['inq_last_6mths']<=2),1,0)
    df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths']>=3)&(df_inputs_prepr['inq_last_6mths']<=6),1,0)
    df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['delinq_2yrs']>6),1,0)
    
    #open_acc
    df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc']==0),1,0)
    df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc']>=1)&(df_inputs_prepr['open_acc']<=3),1,0)
    df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc']>=4)&(df_inputs_prepr['open_acc']<=12),1,0)
    df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc']>=13)&(df_inputs_prepr['open_acc']<=17),1,0)
    df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc']>=18)&(df_inputs_prepr['open_acc']<=22),1,0)
    df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc']>=23)&(df_inputs_prepr['open_acc']<=25),1,0)
    df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc']>=26)&(df_inputs_prepr['open_acc']<=30),1,0)
    df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc']>=310),1,0)
    
    #pub_rec
    df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec']>=0)&(df_inputs_prepr['pub_rec']<=2),1,0)
    df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec']>=3)&(df_inputs_prepr['pub_rec']<=4),1,0)
    df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec']>=5),1,0)
    
    #total_acc
    df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc']<=27),1,0)
    df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc']>=28)&(df_inputs_prepr['total_acc']<=51),1,0)    
    df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc']>=52),1,0)
    
    #acc_now_delinq
    df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq']==0),1,0)
    df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq']>=1),1,0)
    
    #total_rev_hi_lim:: total revenue high limits
    df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim']<=5000),1,0)
    df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim']>5000)& (df_inputs_prepr['total_rev_hi_lim']<=10000),1,0)
    df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim']>10000)& (df_inputs_prepr['total_rev_hi_lim']<=20000),1,0)
    df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim']>20000)& (df_inputs_prepr['total_rev_hi_lim']<=30000),1,0)
    df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim']>30000)& (df_inputs_prepr['total_rev_hi_lim']<=40000),1,0)
    df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim']>40000)& (df_inputs_prepr['total_rev_hi_lim']<=55000),1,0)
    df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim']>55000)& (df_inputs_prepr['total_rev_hi_lim']<=95000),1,0)
    df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim']>95000),1,0)
    
    #annual_inc
    df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc']<= 200000),1,0)
    df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc']> 200000)&(df_inputs_prepr['annual_inc']<= 300000),1,0)
    df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc']> 300000)&(df_inputs_prepr['annual_inc']<= 400000),1,0)
    df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc']> 400000)&(df_inputs_prepr['annual_inc']<= 500000),1,0)
    df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc']> 500000)&(df_inputs_prepr['annual_inc']<= 600000),1,0)
    df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc']> 600000)&(df_inputs_prepr['annual_inc']<= 700000),1,0)
    df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc']> 700000)&(df_inputs_prepr['annual_inc']<= 800000),1,0)
    df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc']> 800000)&(df_inputs_prepr['annual_inc']<= 900000),1,0)
    df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc']> 900000)&(df_inputs_prepr['annual_inc']<= 1000000),1,0)
    df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc']> 1000000)&(df_inputs_prepr['annual_inc']<= 1200000),1,0)
    df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc']> 1200000)&(df_inputs_prepr['annual_inc']<= 1400000),1,0)
    df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc']> 1400000),1,0)
    
    #dti
    df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti']<=1.4),1,0)
    df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti']>1.4)&(df_inputs_prepr['dti']<=3.5),1,0)
    df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti']>3.5)&(df_inputs_prepr['dti']<=7.7),1,0)
    df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti']>7.7)&(df_inputs_prepr['dti']<=10.5),1,0)
    df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti']>10.5)&(df_inputs_prepr['dti']<=16.1),1,0)
    df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti']>16.1)&(df_inputs_prepr['dti']<=20.3),1,0)
    df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti']>20.3)&(df_inputs_prepr['dti']<=21.7),1,0)
    df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti']>21.7)&(df_inputs_prepr['dti']<=22.4),1,0)
    df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti']>16.1)&(df_inputs_prepr['dti']<=20.3),1,0)
    df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti']>35),1,0)
    
    #mths_since_last_delinq
    df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq']>=0) &(df_inputs_prepr['mths_since_last_delinq']<=3),1,0)
    df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq']>=4) &(df_inputs_prepr['mths_since_last_delinq']<=30),1,0)
    df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq']>=31) &(df_inputs_prepr['mths_since_last_delinq']<=56),1,0)
    df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq']>=57),1,0)
    
    #mths_since_last_record
    df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record']>=0) &(df_inputs_prepr['mths_since_last_record']<=2),1,0)
    df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record']>=3) &(df_inputs_prepr['mths_since_last_record']<=20),1,0)
    df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record']>=21) &(df_inputs_prepr['mths_since_last_record']<=31),1,0)
    df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record']>=32) &(df_inputs_prepr['mths_since_last_record']<=80),1,0)
    df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record']>=81) &(df_inputs_prepr['mths_since_last_record']<=86),1,0)
    df_inputs_prepr['mths_since_last_record:>=87'] = np.where((df_inputs_prepr['mths_since_last_delinq']>=87),1,0)
    
    counts += 1

#save processed data
loan_data_inputs_train.to_csv('loan_data_inputs_train.csv')
loan_data_targets_train.to_csv('loan_data_targets_train.csv')
loan_data_inputs_test.to_csv('loan_data_inputs_test.csv')
loan_data_targets_test.to_csv('loan_data_targets_test.csv')   

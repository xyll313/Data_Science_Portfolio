# Credit Risk Modelling

This project build a model to predict personal Probability of Default(PD), Loss given Default (LGD) and Explosure at Default (EAD) using **Logistic regression** (PD) and **beta regression** (LGD&EAD)

The dataset contains all available data for more than 800,000 consumer loans issued from 2007 to 2015 by Lending Club: a large US peer-to-peer lending company. There are several different versions of this dataset. We have used a version available on kaggle.com, which is available at:
: https://www.kaggle.com/wendykan/lending-club-loan-data/version/1

## Results and Discussions

## Conclusions

## Methodology
* **1_Data_Cleaning.ipynb**: 

The data is first cleaned (Data manipulation using **Pandas**) to remove undesired texts for continuous variables, this incldues tidying up of datetime (pd.to_datetime). We then **created dummie variables** for discrete variables as it's the standard procedure for PD model creations. As a last step, we check for missing values (df.isnull().sum()) and filled them with the average of availble data or 0 (depending on how many data points are missing as well as how conservative we want the model to be). 

* **2_Data_Preparation.ipynb**: 

First we create our dependent variable, probability of default. The assumption is if 'loan_status' fall into the follwing categories (**Charge Off, Late (31-120 days), Default, Does not meet the credit policy. Status:Charged Off**), then the loan is defaulted.    

The dataset is then splitted into train and test groups (sklearn). We use train data to build our PD model and the test data would be used later to verify our model. Discrete variables that have similar **Weight of Evidence** are grouped together to reduce the number of features used for predictions. Likely, we disect each continuous variables into a number of dummie variables and group the dummies according to the same principles.

A Python file that does the data cleaning is available: data_prep.py

* **3_PD_Model.ipynb**: 

We biuld the PD Model by **Logistic Regression** and then apply the model on test dataset to calculate **predicted probability of default**. The accuracy of our model is examined by area under ROC (receiver operating characteristic) curve and model performance is evaluated by **Gini coefficient** and **Kolmogorov-Smirnov coefficient**. 

* **4_Score_Card_PD_Model.ipynb**: 

A score card is created based on our PD model. The avaiable minimum and maximum scores are set to be 300 and 850 respectively. Scores for each dummie variable is calculated as, where max_sum_coeff is the sum of maximum coefficients for each category (The coefficients are calculcated by **logistic regression**):

$$variable Score = variable Coef \frac{(max Score-min Score)}{(max Sum Coeff-min Sum Coeff)}$$

Cut-offs for the PD model pre-determines the percentage of customers that would be approved/rejected. We used ROC curve to determine cut off rates

* **5_LGD_Model.ipynb**:

The distribution for recovery rate is as follow and therefore we decided to use a two-stage process
![image](https://user-images.githubusercontent.com/29717509/184869536-81abcf46-7574-4fea-af89-04a1aabde3c1.png)

(1) A logistic regression to see whether the recovery rate is greater than 0
(2) Use a multivariable linear regression model to model recovery rate

* **6_EAD_Model.ipynb**:

The distribution for CCF (the proportion of the original amount of the loan that is still outstanding at the moment when the borrower defaulted)  is as follows:
the proportion of the original amount of the loan that is still outstanding at the moment when the borrower defaulted

![image](https://user-images.githubusercontent.com/29717509/184873461-afc5326e-e420-4696-bdaa-9c515356fd86.png)

Therefore we decided to use only a linear regression model 


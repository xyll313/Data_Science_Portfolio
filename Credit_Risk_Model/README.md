# Credit Risk Modelling

This project build a model to predict Expected Loss (EL) by building models on personal Probability of Default(PD), Loss given Default (LGD) and Explosure at Default (EAD) using **Logistic regression** (PD) and linear regression that approximates **beta regression** (LGD&EAD)

Expected Loss (EL) is defined as: EL = PD x LGD x EAD

The dataset contains all available data for more than 800,000 consumer loans issued from 2007 to 2015 by Lending Club: a large US peer-to-peer lending company. There are several different versions of this dataset. We have used a version available on kaggle.com, which is available at:
: https://www.kaggle.com/wendykan/lending-club-loan-data/version/1

## Results and Discussions
### PD Model 
We created **124 dummie variables** based on their weight of evidence from discrete and continuous independent variables in the dataset to build our PD model using Logistic Regression. 

$$ \frac{P(Y=1)}{P(Y=0)} = \exp{(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 ....)} $$

The model is biuld on the **train_data_set** which contains 80% of the avaiable data. For model evaluation, we plotted ROC(receiver operating chracteristic) curve for AUROC (area under ROC) calculation on the test_data_set, which consists 20% of the available data. 

![image](https://user-images.githubusercontent.com/29717509/184929943-f62b2d57-65b9-4b3b-8701-24996008e71e.png)    
**Figure 1**. ROC plot of our PD model

ROC is a plot of True Positive Rate (TPR) against False positive Rate (FPR). We recall:    
True Positive Rate (TPR) : $TPR = \frac{TP}{TP+FN}$    
False positive Rate (FPR): $FPR = \frac{FP}{FP+TN}$

ROC plot illustrates how TPR and FPR varies as the discrimination threshold (a chosen threshold for calculated probability, values above it assigned as 'default') varies. AUROC give us some information about model accuracy. The calculated **AUROC** for our PD model is **0.700**, which as an AUROC considered poor or fair. However, due to the multivariable and simple nature of our model, an AUROC value at 0.7 is regarded as acceptable. 

Also calculated for model evalucation is Gini coefficient and Kolmogorov-Smirnov Coefficient.

#### 1. Gini Coefficient
Gini Coefficient meausres the statistical dispertion intended to represent the income inequality or the wealth inequality within a nation or a social group.

It defines as: $Gini = 2 \times AUROC - 1$

![image](https://user-images.githubusercontent.com/29717509/184932567-cd1b5c35-5d1f-487f-957b-d7bb7955c407.png)    
**Figure 2**. Gini Coefficient as the area shown in red

In the project, it is used to calculate the level of inequality between non-defaulted and defaulted borrower in a population. **A larger value** indicates a better model. Our model has a Gini Coefficient of **0.574** which is fair.

#### 2. Kolmogorov-Smirnov Coefficient
Kolmogorov-Smirnov Coefficient is the maxium difference between the cumulative distribution functions shows how the model separates non-default and default borrowers. The **greater** the difference, the better the model.

![image](https://user-images.githubusercontent.com/29717509/184933354-71434b7b-b82f-415d-b92f-015cf013eba7.png)    
**Figure 3**  Kolmogorov-Smirnov Test

As we see from the graph, the two cumulative distribution functions are sufficiently far away with a Kolmogorov-Smirnov coefficient of **0.300**. Hence the model has satisfactory predictive power.

### PD Model Score Card
Based on our PD model we build a score-card (df_scorecard.csv) with available score between 3000 and 850.

![image](https://user-images.githubusercontent.com/29717509/184935572-78f7a220-481c-4666-a97d-98424c430115.png)   
**Figure 4**  Scores and approve rate

Each score is associated with a pre-determined approval rate. Therefore the bank can pick a cut-off score with a desired threshold according to its business strategy. 

# LGD Model 
Loss Given Default (LGD) is defined by the percentrage of the exposure that was lost after the borrower defaulted.Our LGD model is built with available defaulted data (*i.e.* the loan has a status of charged off).    
$$LGD = 1 - recovery\:rate$$ where *recovery_rate* is calculated by $['recoveries'] \div ['funded\:amount']$

Since for most defaulted customers, ['recoveres'] equals to zero hence we decided to use a two step approach:    
(1) A logistic regression to see whether the recovery rate is greater than 0        
(2) Use a multivariable linear regression model to model recovery rate

![image](https://user-images.githubusercontent.com/29717509/184938309-89fcdf31-8878-40b9-bc87-85cde2606b95.png)    
**Figure 5** ROC plot for the logistic regression part
For the logistic regression part, AUROC calculated was 0.656, which means the model is fair.

For the linear regression part, we see a correlation at 0.287, which is satisfactory for a LGD model. We can also look at our model erro by plotting the difference in recovery rate between test data (actual) and predicted data:

![image](https://user-images.githubusercontent.com/29717509/184938649-c76dd831-52e4-4096-8efc-4bcbed76c5ce.png)    
**Figure 6** model error: the difference in recovery rate between test data (actual) and predicted data

The model error re-sembles a normal distribution and distributed around zero, which indicates the a good reliability of our model.

# EAD Model
Explosure At Default (EAD) is defined by the amount of the exposure at the moment the borrower default:
$$EAD = ['funded_amount'] * CCF$$
where CCF (credit conversation factor) is defined as the amount of the loan that is still outstanding when the borrower default. Linear regression is chose to model CCF according to its distribution in available data.

![image](https://user-images.githubusercontent.com/29717509/184941588-83448b38-bdf9-4dab-a090-1cfe866d4fdb.png)    
**Figure 7** Model error in CCF

For model error see an approximate normal distribution centred at zero, which means the model's predictive power is relatively good and seems to be a good model methodologically. 
 
## Conclusions
Our model predicted an EL at **8.07%**. Expected loss should be lower than a bank's capital reserve (ECB's requirement currently lies at 15.1% for risk-weighted assets). In general, the expected loss is between 2-10%. Based the real and calculated EL, the bank can adjust it's strategy and decide whether to be more assertive or consertive in giving out loans in the future.

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

* **7_Expected_Loss.ipynb**:
Calculate expected loss by applying: EL = PD x LGD X EAD


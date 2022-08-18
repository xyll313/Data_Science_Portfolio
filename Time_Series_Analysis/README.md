# Introduction

The project works on stock closing price data in four markets (which are time series) obtained by yfiance API. 

# Contents
* **1_Pre_Processing.ipynb**:Data cleaning of time series data. Use pd_to_datetime to turn 'date' into Datetime object; then set date as index(set_index) and finally setting the time series frequencyi (asfreq) as desired, here we used business days (b)

* **2_Introduction.ipynb **:Here we introduced 2 types of special time series, white noise and random walk; 2 features for time series, stationality and seasonality. In the end we illustrated how to calculate **autocorrelation (ACF)** and **partial autocorrelation (PACF)** with statsmodels package. 

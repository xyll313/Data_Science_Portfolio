# Introduction

The project works on stock closing price data in four markets (which are time series) obtained by yfiance API. 

# Contents
* **1_Pre_Processing.ipynb**: Data cleaning of time series data. Use pd_to_datetime to turn 'date' into Datetime object; then set date as index(set_index) and finally setting the time series frequencyi (asfreq) as desired, here we used business days (b)

* **2_Introduction.ipynb**: Here we introduced 2 types of special time series, white noise and random walk; 2 features for time series, stationality and seasonality. In the end we illustrated how to calculate **autocorrelation (ACF)** and **partial autocorrelation (PACF)** with statsmodels package. 

* **3_AR_Model.ipynb**: AR model is a linear model, where current period values are a sum of of past outcomes multiplied by a numeric factor:
$$x_t = C + \phi x_{t-1} + \epsilon_t$$
AR model works best with stationary data.
FTSE price data are processed by an AR Model. We used PACF to determine how many lags to include in our model. 
We looked at Price (which is not stationary), Returns, Normalised Returns and residual (white noise).

* **4_MA_Model.ipynb**: MA model is with marginal intigration and we use ACF to determine the number of lags required
$$x_t = C + \theta_1 \epsilon_{t-1} + \epsilon_{t}$$

* **5_ARMA_Model.ipynb**: AR + MA model
$$x_t = C + \phi x_{t-1} +  \theta_1 \epsilon_{t-1} + \epsilon_{t}$$

* **6_ARIMA_Model.ipynb**:In addition to an ARMA(p,q) model, an ARIMA (p,d,q) model also differencing the original series d times (and hence would lose d observations)

* **7_ARCH_Model.ipynb**:Autoregressive Conditional Heteroskedasticity Model. It is used to measure volatility
For a simple ARCH(1, include only one previous value) model, we have:
$$Var(y_t|y_{t-1}) =  \sigma^2_{t} = \alpha0 + \alpha_1 \epsilon_{t-1}^2$$
where $\alpha_1$ is approximate to \theta_1 in ARIMA models. 

At higher orders, we have: $\sigma^2_{t} = \alpha0 + \alpha_1 \epsilon_{t-1}^2 + \alpha_2 \epsilon_{t-2}^2$

$$y_t = \mu_t + \epsilon_t$$ 
where $\mu$ is the mean, $\mu_t = C + \phi_1 \mu_{t-1}$ and variance as shown above

---
title: "Machine Learning-Driven Multi-Timeframe Trading Strategy for SPY ETF"
author: "Ayan Goswami"
date: "2025-07-16"
output:
  pdf_document:
    toc: true
    toc_depth: 3
  html_document:
    toc: true
    toc_float: true
    theme: flatly
---

# Abstract

This paper presents a machine learning framework for intraday trading of the SPDR S&P 500 ETF (SPY) using standardized formats of traditional indicators. We employ logistic regression, XGBoost, and support vector machines (SVM) to predict short-term profitability. The strategies leverage Average True Range (ATR), Moving Averages, and Volume to try and predict short-term momentum bursts (MBs). This paper will focus on long-only strategies as they are less risky when implementing in a production environment. Emphasis is placed on high precision and minimizing false positives using probabilistic thresholds derived from model outputs. Backtests demonstrate that our methods significantly outperform random strategies in terms of Sharpe ratio, precision, and drawdown control.

# Introduction

The development of statistically sound and precision-driven trading strategies is essential in modern algorithmic finance. This paper addresses the problem of directional prediction of SPY ETF trades using machine learning and statistical modeling over multi-timeframe indicators. We aim to optimize execution precision rather than naive accuracy, focusing on reducing false positive trade signals.

# Data and Features
The dataset comprises one-minute interval price data (OHLCV: Open, High, Low, Close, Volume) for the SPDR S&P 500 ETF Trust (SPY) spanning from May 29th, 2023, to July 17th, 2023. From these primary market data points, we derived a set of technical indicators commonly employed in financial analysis. While these indicators typically involve parameterization, we adopted industry-standard parameters to maintain methodological consistency and prevent overfitting through excessive parameter optimization. This research focuses primarily on optimizing machine learning models' predictive capabilities using these established indicators, rather than engaging in indicator parameter optimization. This approach allows us to isolate the effectiveness of various machine learning architectures while maintaining the integrity of widely-accepted technical analysis frameworks.

## Timeout (t)
There is no right way to time price trends, hence the best path forward to specialize in different time frames is to set an automatic timeout for trades. This way we categorize different types of momentum bursts based on their time period, and each strategy occupies its own seperate area of specialization. Thus, we introduce 5 primary timeout periods for this paper:
1. 3 minutes
2. 5 minutes
3. 15 minutes
4. 60 minutes
5. 180 minutes

## ATR (volatility measure)

ATR (Average true range) is a technical indicator that measures market volatility by decomposing the entire range of an asset's price for a given period. Developed by J. Welles Wilder, it's calculated using the previous price movements of the security. For the purpose of our analysis, it was calculated using standard deviation.

$$ ATR_T = \sqrt{\frac{\sum_{i=T-k}^{T}(OHLC_i - \mu_T)^2}{k}} $$ 


Where $OHLC = \frac{Open+High+Low+Close}{4}$ and $k$ is the look back period. The look back period standard in the industry is 14, and we set this as our look back period as well.

The motivation was to control
the sensitivity to market volatility and potentially use as a regressor. The purpose is that it scales profit/loss targets based on current market conditions

During the data mining process, we introduced an ATR scaling factor $\lambda$ which uses ATR to increase profit and loss margins. 
- **Example**: If SPY ATR = 0.50 and $\lambda = 10$:
    - Base volatility adjustment = 10 × 0.50 = \$5.00
    - Hypothetically: Sell if SPY is up 5$ from entry price

For longer time periods, the price might fluctuate a lot more than shorter time frames. In order to prevent being stopped out of a potentially profitable but volatile trade, the bounds for profit and loss are set a lot higher in trading windows with higher timeouts. In order to determine what the multiplier should be for the defined timeouts we implement a logarithmic scaling function that increases position bounds proportionally to the trading window duration.

The relationship between timeframe and volatility follows a logarithmic pattern, where:

$$\lambda = 4\log(t-2)+1$$

Hence the relationships for each timeout are as follows:

1. 3 minutes $t$ => $\lambda = 1$
1. 5 minutes $t$ => $\lambda = 3$
1. 15 minutes $t$ => $\lambda = 5$
1. 60 minutes $t$ => $\lambda = 8$
1. 180 minutes $t$ => $\lambda = 10$

Furthermore, we introduced an asymmetric risk-ratio ($\chi$) which determines how much higher the take profit price will be compared to the stop loss, for a given $\lambda$. For simplicity though, this was fixed to 1.5. Further exploration can be conducted to fine tune this parameter

Our `pl_value` column was determined by a 1000 shares of SPY, where the targets were:
- **Profit Target** $PT$: $Price + \lambda * \chi × ATR$

- **Loss Target** $LT$: $Price - \lambda * ATR$

For each entry point $i$, the algorithm checks future prices within time window $t$ minutes. This is done pessimistically, where the stop loss is evaluated first:
Stop Loss Check:
```
if Low_{i+j} ≤ Loss_target:
    PL = 0  # Loss trade
    pl_value = (Loss_target - Open_i) × 1000
where:
Loss_target = Open_i - (λ × ATR_i)
j ∈ [0, t]
```

Take Profit Check:
```
if High_{i+j} ≥ Profit_target:
    PL = 1  # Winning trade
    pl_value = (Profit_target - Open_i) × 1000
where:
Profit_target = Open_i + (λ × ATR_i)
j ∈ [0, t]
```
Time Expiry:
```
if j = t and no target hit:
    pl_value = (Close_{i+t} - Open_i) × 1000
```
The `PL` column was used for encoding a binary win/lose variable for the machine learning models, and `pl_value `was used to determine drawdown.

## Standardized Moving Average

Moving averages are very common in technical analysis, and they smooth price data by calculating the average price over a specified period, creating a trend-following indicator. Since these are always tied to the price of the security, we standardize it using the following formula: 


$$SMA_k = \frac{Open_T}{\frac{1}{k}\sum_{i=T-k}^{T-1}Close_i}$$

-   $T$ = current time period

-   $k$ = lookback period

-   Hypothetically, values \> 1.0 indicate price above historical
    average (bullish)
-   Values \< 1.0 indicate price below historical average (bearish)

Multiple Moving Averages are often combined and used in tandem to contextualize market trends. We try and capture this relationship by introducing interactions terms in our models. The $k$'s chosen for this analysis are:
- SMA\_7
- SMA\_20
- SMA\_50


## Normalized trading volume

We normalized the volume due to the extremely volatile nature of the SPY intra-day market. To help reduce the noise we applied the following transformation.

$$norm\_volume = \frac{E(V) - V}{\sqrt{Var(V)}}$$



# Methodology

## Logistic Regression

Logistic regression models the log-odds of profitability as a linear function:

$$
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_k X_k
$$

Variable selection via AIC and Lasso was performed. Thresholds were tuned using normal quantiles to maximize precision.

## XGBoost

XGBoost was trained to maximize AUC-PR, reflecting focus on precision:

$$
L(\phi) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)
$$

## Support Vector Machines

We used an RBF kernel SVM to detect nonlinear boundaries. Thresholding was done similarly to GLM and XGBoost.

# Evaluation Metrics

- **Sharpe Ratio**: Mean return over return standard deviation
- **Max Drawdown**: Largest capital loss from peak
- **Precision**: $TP / (TP + FP)$

# Results

| Model | Sharpe | Max Drawdown | Precision |
|-------|--------|---------------|-----------|
| GLM | 6.27 | -2.68% | ~51.7% |
| XGBoost | 16.81 | 0% | 83.2% |
| SVM | 0.46 | 0% | 80.6% |
| Ensemble | 11.77 | -1.92% | 100% (filtered) |

# Backtest Visuals

_Equity curves for each model are included in the full PDF version._

# Discussion

SVMs and tree-based models outperform linear methods in non-linear regions, but logistic regression showed the best live performance due to generalization. Our ensemble strategy confirms that agreement between diverse models improves confidence in trade execution.

# Conclusion

This study presents a robust ML framework for SPY trading. By prioritizing precision and leveraging ensemble thresholding, we achieve statistically significant results exceeding random baselines.

# References

- Agresti, A. (2013). *Categorical Data Analysis*. 3rd ed.
- Chen, T., & Guestrin, C. (2016). XGBoost.
- Vapnik, V. (1995). *The Nature of Statistical Learning Theory*.

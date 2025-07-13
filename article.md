# Multi-Timeframe Algorithmic Trading: A Machine Learning Approach to SPY ETF Trading

## Abstract

This paper introduces the 2023-momentum system, an algorithmic trading framework developed to trade the SPY ETF using ensemble machine learning methods. The system leverages predictive models operating across multiple timeframes—specifically 1-3 minutes, 3-5 minutes, and 5-15 minutes—to identify optimal trading opportunities. Its core innovations include adaptive Average True Range (ATR)-based risk management, multi-model ensemble predictions, and a robust real-time execution environment. Extensive backtesting demonstrates consistent profitability and controlled drawdowns, indicating resilience across varied market conditions.

## 1. Introduction

Algorithmic trading has witnessed significant advancements through the integration of machine learning (ML) methodologies. This study presents a systematic approach utilizing multiple ML models to trade the SPY ETF, with the goal of effectively capturing short-term price fluctuations while systematically managing risk. Motivated by the complexity of intraday market dynamics, the 2023-momentum system addresses critical challenges including simultaneous management of signals across various timeframes, rigorous risk management grounded in market volatility, the amalgamation of diverse ML models for predictive accuracy, and low-latency trade execution suitable for intraday trading strategies.

## 2. Background and Motivation

The impetus for developing the 2023-momentum system arises from the necessity to systematically exploit short-term pricing inefficiencies in the SPY ETF. Conventional technical analysis methods often lack sufficient analytical rigor for sustained profitability, whereas purely machine learning-based strategies may neglect critical microstructural market effects. Ensemble methods offer a robust solution by capitalizing on the strengths of various predictive models. For instance, Random Forest effectively handles complex, nonlinear market relationships; Support Vector Machines (SVMs) provide clear classification boundaries; K-Nearest Neighbors (KNN) excels at detecting localized patterns; and linear models offer interpretability and computational speed. Recognizing the multifaceted nature of market dynamics, the system integrates these models across multiple timeframes.

## 3. Methodology

### 3.1 Data Processing Pipeline

The system processes one-minute SPY data using a specialized FinData class responsible for data normalization, cleaning, and technical indicator computation. The implemented indicators include an adjusted ATR, computed as a rolling price standard deviation:

$$ATR_t = \sqrt{\frac{\sum_{i=t-k}^{t}(P_i - \mu_t)^2}{k}}$$

and a standardized moving average (SMA), expressing current prices relative to historical averages:

$$SMA_k = \frac{P_t}{\frac{1}{k}\sum_{i=t-k}^{t-1}P_i}$$

This normalization technique ensures the indicators adapt effectively to shifting market conditions.

### 3.2 Feature Engineering

An extensive set of engineered features comprises normalized SMA ratios, ATR-based volatility metrics, momentum indicators, and cross-timeframe technical signals. This comprehensive approach ensures robust model performance by capturing various market dynamics and conditions.

### 3.3 Model Architecture

The employed ML architecture involves distinct implementations of Random Forest, SVM, and other methods. Specifically, Random Forest models were trained for both classification and regression tasks, while SVMs utilized radial basis function (RBF) kernels to capture nonlinear market behaviors alongside linear kernels for baseline comparisons. Multiple strategies were developed to function across different timeframes: the ultra-short-term momentum capture strategy (1-3 minutes), the short-term trend-following strategy (3-5 minutes), and the medium-term position-holding strategy (5-15 minutes).

### 3.4 Risk Management Framework

Risk management utilizes dynamically calculated profit-taking and stop-loss levels anchored on the ATR metric:
$$TP = P_{entry} + \alpha \cdot \chi \cdot ATR_t$$
$$SL = P_{entry} - \chi \cdot ATR_t$$

where $\alpha$ represents the profit multiplier (typically 1.5), and $\chi$ is the strategy-specific risk parameter. This framework systematically manages risk exposure and profit realization, allowing strategic adaptation to varying market volatility.

## 4. Experimental Design

### 4.1 Backtesting Methodology

Backtesting was conducted using a pragmatic three-fold dataset split, maintaining a realistic separation between training and testing periods. Realistic transaction costs and slippage were integrated into performance assessments to ensure practical relevance.

### 4.2 Performance Metrics

The evaluation of strategies utilized key performance indicators including Sharpe ratio:


$$
Sharpe\ Ratio = \frac{R_p - R_f}{\sigma_p}
$$

maximum drawdown:

$$
Maximum\ Drawdown = \max_{t \in [0,T]} \frac{Peak_t - Trough_t}{Peak_t}
$$


win rate, and profit factor, providing a comprehensive picture of model effectiveness.

## 5. Results and Analysis

Detailed backtesting across distinct timeframes highlighted each strategy’s strengths. The 1-3 minute strategy yielded optimal results from KNN models, whereas Random Forest excelled in the 3-5 minute strategy, offering superior risk-adjusted returns. The medium-term 5-15 minute strategy demonstrated effectiveness primarily through decision tree and GLM-based trend reversal identification. Live trading results corroborated backtesting performance, exhibiting sustained profitability, effective risk mitigation, and minimal execution slippage.

### 5.3 Statistical Significance

The strategies achieved statistically significant performance metrics, with robust t-statistics, information ratios, and Calmar ratios, underscoring their reliability and effectiveness.

## 6. System Architecture

### 6.1 Real-Time Execution Engine

The production system implements:

python
class Trade:
    def __init__(self, profit_price, loss_price, entry_price, quantity, strategy, timeout):
        self.profit_price = profit_price
        self.loss_price = loss_price
        self.entry_price = entry_price
        self.timeout = timeout
        self.quantity = quantity
        self.strategy = strategy


### 6.2 Model Integration and Signal Generation

The system uses R for model inference with Python for execution. Signal thresholds were determined through backtesting optimization:

r
# Load trained models
lm_1_3 <- readRDS("./models/SPY_1m/SPY_1m_lm_1_3.rds")
glm_3_5 <- readRDS("./models/SPY_1m/SPY_1m_glm_3_5.rds")
glm_5_15 <- readRDS("./models/SPY_1m/SPY_1m_glm_5_15.rds")

# Generate predictions
pred_1_3 <- predict(lm_1_3, current_data)
pred_glm_3_5 <- predict(glm_3_5, current_data, type="response")
pred_glm_5_15 <- predict(glm_5_15, current_data, type="response")

# Apply optimized thresholds
bullish_1_3 <- ifelse(pred_1_3 > 0.8634, 1, 0)
bullish_3_5 <- ifelse(pred_glm_3_5 > 0.5914 & pred_lm_3_5 > 1.558, 1, 0)
bullish_5_15 <- ifelse(pred_glm_5_15 > 0.5146, 1, 0)


These thresholds balance precision and recall based on historical performance analysis.

### 6.3 Infrastructure Components

- **Data Pipeline**: Real-time price feed integration
- **Model Serving**: Containerized R model inference
- **Execution Engine**: Python-based order management
- **Risk Monitor**: Continuous position and exposure tracking
- **Logging System**: Comprehensive audit trail

## 7. Risk Analysis and Mitigation

## 7. Practical Implementation Challenges

### 7.1 Data Quality and Processing

Real-time market data presents several challenges:
- **Missing Data**: Handling gaps in price feeds during low-volume periods
- **Outliers**: Filtering erroneous price spikes that could trigger false signals
- **Latency**: Ensuring indicators are calculated quickly enough for real-time decisions

The FinData class addresses these issues through robust data validation and efficient calculation methods.

### 7.2 Model Deployment

Integrating R models with Python execution required careful consideration:
- **Serialization**: Models saved as .rds files for consistent loading
- **Threshold Optimization**: Signal thresholds determined through extensive backtesting
- **Performance Monitoring**: Tracking prediction accuracy in real-time

### 7.3 Risk Control Implementation

Practical risk management goes beyond theoretical formulas:
- **Position Limits**: Hard caps on trade size regardless of model confidence
- **Correlation Monitoring**: Preventing over-concentration in similar trades
- **Emergency Stops**: Manual override capabilities for unusual market conditions

## 8. Performance Attribution

### 8.1 Alpha Generation Sources

Primary sources of excess returns:
- **Mean Reversion**: Short-term price inefficiencies
- **Momentum Capture**: Trend-following in favorable conditions
- **Volatility Timing**: ATR-based position sizing optimization
- **Ensemble Benefits**: Model diversification effects

### 8.2 Transaction Cost Analysis

Comprehensive cost analysis reveals:
- **Bid-Ask Spreads**: Minimal impact due to SPY liquidity
- **Commission Costs**: Fixed costs amortized across volume
- **Market Impact**: Negligible for typical position sizes
- **Slippage**: Well-controlled through limit order usage

## 9. Future Enhancements

### 9.1 Advanced Machine Learning

Potential improvements include:
- **Deep Learning**: LSTM networks for sequence modeling
- **Reinforcement Learning**: Adaptive strategy optimization
- **Feature Learning**: Automated technical indicator discovery
- **Transfer Learning**: Cross-asset model adaptation

### 9.2 Alternative Data Integration

Expansion opportunities:
- **Sentiment Analysis**: Social media and news sentiment
- **Options Flow**: Derivative market signals
- **Economic Indicators**: Macro-economic factor integration
- **Cross-Asset Signals**: Multi-market correlation analysis

### 9.3 Infrastructure Scaling

System enhancements:
- **Cloud Deployment**: Scalable compute resources
- **Multi-Asset Support**: Portfolio-level optimization
- **Real-Time Analytics**: Enhanced monitoring capabilities
- **API Integration**: Broader broker connectivity

## 10. Lessons Learned and Future Work

Developing 2023-momentum provided valuable insights into practical algorithmic trading:

**What Worked Well**:
- Multi-timeframe approach captured different market dynamics effectively
- ATR-based risk management adapted well to changing volatility
- Ensemble methods provided more robust predictions than individual models
- The 3-5 minute strategy found the sweet spot between signal quality and execution frequency

**Key Challenges**:
- Model overfitting required careful validation procedures
- Real-time execution introduced latency considerations not present in backtesting
- Market regime changes occasionally reduced model effectiveness
- Transaction costs and slippage impact profitability more than initially expected

**Future Improvements**:
- Incorporate regime detection to adapt model weights dynamically
- Add alternative data sources like options flow or sentiment indicators
- Implement reinforcement learning for adaptive position sizing
- Expand to other liquid ETFs for diversification

This project demonstrates that systematic approaches to trading can be profitable when properly implemented with appropriate risk controls. The key is combining sound statistical methods with practical market knowledge and robust execution infrastructure.

## References and Resources

**Technical References**:
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Chan, E. (2009). Quantitative Trading: How to Build Your Own Algorithmic Trading Business
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning

**Data Sources**:
- Market data obtained through standard financial APIs
- Technical indicators calculated using custom implementations
- Backtesting performed on historical SPY data from 2022-2023

**Tools and Libraries**:
- R: randomForest, e1071, class, rpart packages
- Python: pandas, numpy for data processing
- AWS Chalice for cloud deployment

## Appendix A: Model Specifications

### A.1 Random Forest Parameters
- Number of trees: 1000
- Features per split: √p (where p is total features)
- Minimum samples per leaf: 5
- Bootstrap sampling: True

### A.2 SVM Configuration
- Kernel types: RBF, Linear
- Regularization parameter C: Grid-searched
- Gamma parameter: Auto-scaled
- Probability estimates: Enabled

### A.3 Feature Set Details
- ATR (Average True Range): 15-period
- SMA periods: 7, 20, 50, 100
- Price ratios: Current/SMA for each period
- Momentum indicators: Multi-period price changes
- Volume-weighted features: Price-volume relationships

## Appendix B: Performance Tables

[redacted]

# 2023-momentum: Advanced Algorithmic Trading System

## Overview

2023-momentum is an algorithmic trading system I developed for SPY ETF trading using machine learning techniques. The system combines multiple models across different timeframes to identify profitable trades while managing risk through dynamic position sizing. What makes this system unique is its multi-timeframe approach and real-time execution capabilities. Refer to article.md, mkt_data.ipynb and **strat_book.pdf** for details.

## Key Features

### 🤖 Machine Learning Models
- **Random Forest**: Used for both classification (profit/loss) and regression (profit amount)
- **Support Vector Machines**: Multiple kernel types tested, RBF performed best
- **K-Nearest Neighbors**: Surprisingly effective for short-term predictions
- **Linear Models**: Provide stable baseline performance and fast inference
- **Decision Trees**: Excel at capturing trend reversals in longer timeframes

### 📊 Trading Strategies
- **1-3 Minute**: Quick momentum plays, high frequency but smaller positions
- **3-5 Minute**: Main strategy with best risk-adjusted returns
- **5-15 Minute**: Longer holds for trend continuation, lower frequency

Each strategy uses different model combinations and risk parameters optimized through backtesting.

### 🎯 Advanced Risk Management
- **Dynamic Stop-Loss**: ATR-based adaptive stop placement
- **Take-Profit Optimization**: Volatility-adjusted profit targets
- **Position Sizing**: Risk-proportional allocation based on market conditions
- **Timeout Protection**: Maximum holding period constraints
- **Drawdown Controls**: Portfolio-level risk monitoring

### ⚡ Real-Time Execution Engine
- **Live Market Data**: 1-minute SPY price feeds with technical indicators
- **Model Inference**: R-based prediction pipeline with Python execution
- **Order Management**: Automated trade execution with comprehensive logging
- **Performance Monitoring**: Real-time P&L tracking and risk metrics

## Technical Architecture

### Data Pipeline
```
Market Data → Feature Engineering → Model Inference → Signal Generation → Trade Execution
```

### Core Components

#### 1. Data Collection (`JSON txt data/`)
- **SPY_1m_data.txt**: High-frequency 1-minute OHLC data
- **SPY_15m_data.txt**: 15-minute aggregated data for validation

#### 2. Model Training (`SPY training/`)
- **Training Datasets**: Multiple timeframe datasets (train1_3.csv, train3_5.csv, etc.)
- **Strategy Development**: R Markdown notebooks with comprehensive backtesting
- **Model Persistence**: Serialized models for production deployment

#### 3. Trading Infrastructure (`Trade infra/`)
- **Execution Engine**: Python-based real-time trading system
- **Model Loading**: R script for inference pipeline
- **Risk Management**: Dynamic position and capital management
- **API Integration**: Chalice-based cloud deployment framework

## Mathematical Framework

### Technical Indicators

The system uses a modified ATR calculation based on price standard deviation:
```
ATR = sqrt(Σ(Price_i - Mean)² / k)
```

Standardized Moving Averages normalize current price against historical averages:
```
SMA_k = Current_Price / Average(Past_k_Prices)
```

This normalization helps the models work across different price levels and market conditions.

### Risk Management Formulas

**Take Profit Level**:
```
TP = P_entry + (1.5 * χ * ATR_t)
```

**Stop Loss Level**:
```
SL = P_entry - (χ * ATR_t)
```

Where χ is the strategy-specific risk parameter.

### Model Ensemble

**Random Forest Prediction**:
```
RF(x) = (1/B) * Σ(T_b(x)) for b=1 to B
```

**SVM Optimization**:
```
min_{w,b,ξ} (1/2)||w||² + C*Σ(ξ_i)
```

## Performance Metrics

### Backtesting Results

| Strategy | Final Capital | Win Rate | Max Drawdown | Sharpe Ratio |
|----------|---------------|----------|--------------|--------------|
| 1-3 Min  | $1,200+      | 65%      | -8.5%        | 1.8          |
| 3-5 Min  | $1,350+      | 72%      | -6.2%        | 2.1          |
| 5-15 Min | $1,280+      | 68%      | -7.8%        | 1.9          |

### What I Learned

**Model Selection Insights**:
- KNN works surprisingly well for very short timeframes (1-3 min)
- Random Forest provides the most consistent results across timeframes
- Decision trees excel at longer timeframes where trends are more established
- Linear models offer good baseline performance with fast execution

**Strategy Development**:
- The 3-5 minute timeframe offers the best balance of signal quality and execution frequency
- Combining classification and regression models improves overall performance
- ATR-based position sizing is crucial for risk management

## Installation & Setup

### Prerequisites
- Python 3.8+
- R 4.0+
- Required Python packages: `pandas`, `numpy`, `chalice`
- Required R packages: `randomForest`, `e1071`, `class`, `rpart`, `caret`

### Installation Steps

1. **Clone Repository**:
```bash
git clone <repository-url>
cd bot2-pvt
```

2. **Install Python Dependencies**:
```bash
pip install -r Trade\ infra/API/requirements.txt
```

3. **Install R Dependencies**:
```r
install.packages(c("randomForest", "e1071", "class", "rpart", "caret"))
```

4. **Configure API Keys**:
```bash
# Update Trade infra/API/chalicelib/token.json with your credentials
```

## Usage

### Local Development

1. **Initialize Price History**:
```bash
cd "Trade infra"
python load_initial.py
```

2. **Start Trading System**:
```bash
python local_trading.py
```

### Cloud Deployment

1. **Deploy API**:
```bash
cd "Trade infra/API"
chalice deploy
```

2. **Monitor Performance**:
```bash
tail -f ../logs.txt
```

## File Structure

```
bot2-pvt/
├── JSON txt data/           # Market data files
│   ├── SPY_1m_data.txt     # 1-minute OHLC data
│   └── SPY_15m_data.txt    # 15-minute OHLC data
├── SPY training/           # Model development
│   ├── csvs/              # Training datasets
│   └── workbooks/         # R analysis notebooks
├── Trade infra/           # Production system
│   ├── API/              # Cloud deployment
│   ├── data/             # Capital tracking
│   ├── models/           # Trained models
│   ├── local_trading.py  # Main execution engine
│   ├── trade_class.py    # Trade management
│   ├── get_price.py      # Data acquisition
│   └── logger.py         # Logging utilities
├── article.md            # Research documentation
└── README.md            # This file
```

## Key Algorithms

### Feature Engineering
- **Price Normalization**: Current price relative to moving averages
- **Volatility Measures**: ATR-based market condition assessment
- **Momentum Indicators**: Multi-period price change calculations
- **Cross-Timeframe Signals**: Aggregated technical indicators

### Signal Generation
```r
# Example signal logic
bullish_1_3 <- ifelse(pred_1_3 > 0.8634, 1, 0)
bullish_3_5 <- ifelse(pred_glm_3_5 > 0.5913871 & pred_lm_3_5 > 1.558466, 1, 0)
bullish_5_15 <- ifelse(pred_glm_5_15 > 0.5146253, 1, 0)
```

### Risk Controls
- **Maximum Position Size**: 10 shares per trade
- **Concurrent Trades**: Multiple strategies can run simultaneously
- **Capital Preservation**: Automatic shutdown on excessive losses
- **Timeout Mechanisms**: Forced exit after maximum holding period

## Monitoring & Logging

### Real-Time Metrics
- **Active Positions**: Current trade status and P&L
- **Capital Tracking**: Strategy-specific performance
- **Risk Exposure**: Portfolio-level risk assessment
- **Model Performance**: Prediction accuracy monitoring

### Log Analysis
```bash
# View recent trading activity
tail -n 100 Trade\ infra/logs.txt

# Monitor capital changes
cat Trade\ infra/data/capital/capital_*.txt
```

## Technical Challenges Solved

**Data Processing**: Built a custom FinData class to handle real-time market data and calculate technical indicators efficiently.

**Model Integration**: Solved the challenge of combining R-based models with Python execution through a clean API interface.

**Risk Management**: Implemented dynamic stop-loss and take-profit levels based on market volatility (ATR).

**Real-time Execution**: Created a robust trading engine that handles multiple strategies simultaneously with proper logging and error handling.

## Performance Optimization

### Model Tuning
- **Hyperparameter Optimization**: Grid search for optimal parameters
- **Feature Selection**: Recursive feature elimination
- **Cross-Validation**: Time-series aware validation techniques
- **Ensemble Weighting**: Dynamic model combination strategies

### Execution Optimization
- **Latency Reduction**: Streamlined prediction pipeline
- **Memory Management**: Efficient data structure usage
- **Error Handling**: Robust exception management
- **Failover Mechanisms**: Backup execution pathways

## Future Enhancements

### Advanced Analytics
- **Deep Learning**: LSTM networks for sequence modeling
- **Reinforcement Learning**: Adaptive strategy optimization
- **Alternative Data**: Sentiment and options flow integration
- **Multi-Asset**: Portfolio-level optimization

### Infrastructure Improvements
- **Cloud Scaling**: Kubernetes deployment
- **Real-Time Analytics**: Enhanced monitoring dashboards
- **API Expansion**: Multiple broker integrations
- **Database Integration**: Historical performance storage

## Contributing

This project demonstrates advanced quantitative finance techniques and serves as a foundation for algorithmic trading research. The modular architecture allows for easy extension and experimentation with new models and strategies.

## Important Notes

This project was developed as part of my quantitative finance studies. The system has been tested extensively in simulation but should be thoroughly validated before any live trading. The code demonstrates practical applications of machine learning in finance and systematic trading approaches.

All performance results are based on historical backtesting and may not reflect future performance. The system includes comprehensive risk controls, but trading always involves risk of loss.

## License

This project is developed for academic and research purposes. Please ensure compliance with all applicable financial regulations and obtain proper licensing before commercial deployment.

---

**Contact**: For questions about the research methodology or system architecture, please refer to the accompanying research article (`article.md`) for detailed technical documentation.
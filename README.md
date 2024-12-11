
# Predicting_Bank_Loan_Default_Using_Machine_Learning


## Overview
This project explores the relationship between news headlines and stock performance, leveraging Natural Language Processing (NLP) and machine learning models to develop a sentiment-driven trading strategy. By analyzing sentiment from headlines and integrating it with financial indicators, the strategy achieved a top return of 75% with Amazon (AMZN) stock.

## Key Highlights
- Dataset: Utilized a dataset containing 9,470 headlines with 5 features from 10 selected stocks.
- Sentiment Analysis: Applied tokenization techniques using NLTK and developed sentiment scores using various models, including a Rule-Based approach (VADER).
- Models Used:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Support Vector Machine (SVM)
  - Long Short-Term Memory (LSTM)
  - VADER (Rule-Based)
- Results: The VADER model demonstrated the highest correlation between sentiment scores and stock returns, achieving optimal trading performance..
- Trading Strategy: Integrated sentiment scores with a moving average-based strategy to generate trading signals.

## Dependencies
To run this project, you'll need the following Python libraries:

- pandas
- numpy
- nltk
- matplotlib
- seaborn
- scikit-learn
- tensorflow (for LSTM)
- vaderSentiment
- yfinance (for stock price data)
## Methodology
* Sentiment Analysis:
  * Headlines were tokenized using NLTK for text preprocessing.
  - Multiple models, including Logistic Regression, KNN, Decision Tree, SVM, and LSTM, were evaluated for sentiment analysis.
  - The VADER model, a rule-based sentiment analysis approach, demonstrated the highest correlation between sentiment scores and stock returns.

* Trading Strategy:
  - The sentiment score from VADER was combined with moving averages to engineer a trading strategy.
  - Stocks with positive sentiment scores were evaluated using a moving average strategy to identify buy/sell signals.
  - Backtested the strategy on historical stock data, generating buy/sell signals based on sentiment and price trends.
## Results

- The VADER model outperformed machine learning models in correlating sentiment with stock returns.
- The sentiment-driven trading strategy achieved:
  - Amazon (AMZN): 75% return
  - Other stocks: Moderate returns depending on sentiment trends.
## Conclusion
This project demonstrates the potential of integrating sentiment analysis with traditional stock market indicators to engineer profitable trading strategies. The VADER model, combined with a moving average strategy, showed significant improvements in stock returns. While the strategy performed exceptionally well on specific stocks like Amazon, further refinement is required for broader applicability.
## Future Improvements

- Test the strategy on a larger dataset with a wider variety of stocks.
- Incorporate additional technical indicators for improved trading signals.
- Experiment with advanced NLP models (e.g., BERT) for sentiment analysis.
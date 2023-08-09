import math as m

import numpy as np

from portfolio.functions import *

'''
Our Portfolio class:
- It calculate and store each kind of data such as total value, repartition, historical prices, volatility, returns, shape ratio and more
- Historical prices are fetched from the alpaca API => it's up to you to implement a new provider if needed. (get_stocks_prices to modify)
- It's the same for the portfolio that we load if we don't give one when creating the class. It's loaded from your alpaca account but could be loaded in an excel file as well.

'''


class Portfolio:

    def __init__(self, api_key='', api_secret_key='', portfolio=None):

        # Defining keys
        self.api_key = api_key
        self.api_secret_key = api_secret_key

        # Loading portfolio
        self.portfolio = self.load_portfolio() if portfolio is None else portfolio

        # Defining variables
        self.weights = np.array(list(map(lambda x: float(x), self.portfolio['market_value'].values))) / sum(
            list(map(lambda x: float(x), self.portfolio['market_value'].values)))
        self.value = np.round(sum(list(map(lambda x: float(x), self.portfolio['market_value'].values))), 2)
        self.historical_data = self.get_historical_data(list(self.portfolio['symbol']))
        self.volatility = np.round(self.get_annualized_volatility(), 2)
        self.returns = self.get_annualized_returns()

        # Calculate sharpe ratio
        self.sharpe_ratio = (self.returns - 0.02) / self.volatility

    def load_portfolio(self):

        # Load portfolio
        url = "https://paper-api.alpaca.markets/v2/positions"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret_key
        }
        data = json.loads(requests.get(url, headers=headers, verify=False).text)
        portfolio = pd.DataFrame(data)
        columns = ['symbol', 'exchange', 'qty', 'side', 'market_value']

        return portfolio[columns]

    def get_historical_data(self, symbols):

        # Get historical data from our provider
        if self.api_key == '' or self.api_secret_key == '':
            portfolio_data = get_stocks_prices_from_yahoo(symbols)

        # If we have an api key, we use alpaca
        else:
            stocks_prices = get_stocks_prices_from_alpaca(symbols, self.api_key, self.api_secret_key)

            # Formatting data
            portfolio_data = pd.DataFrame(columns=['date'])
            for stock in stocks_prices:
                stocks_prices[stock].columns = ['date', stock]
                stocks_prices[stock][stock] = stocks_prices[stock][stock].pct_change()
                portfolio_data = pd.merge(portfolio_data, stocks_prices[stock], on='date', how='outer')
            portfolio_data = portfolio_data.fillna(0).set_index('date')

        return portfolio_data

    def get_annualized_volatility(self):

        # Get annualized volatility
        # returns = self.historical_data.fillna(0).dot(self.weights.T)
        covariance = np.cov(self.historical_data.fillna(0).T)
        portfolio_volatility = np.sqrt(np.dot(np.dot(self.weights, covariance), self.weights.T)) * m.sqrt(252)

        return portfolio_volatility

    def get_annualized_returns(self):

        # Get annualized average return over the YTD
        return (1 + self.historical_data.fillna(0).dot(self.weights.T)).prod() - 1

    def get_indicators(self):

        # Get moving average indicators
        indicators = {}
        for column in self.historical_data.columns:
            temp_df = self.historical_data[[column]].copy()
            temp_df['ma1'] = (1 + temp_df[column]).cumprod().rolling(10).mean()
            temp_df['ma2'] = (1 + temp_df[column]).cumprod().rolling(30).mean()
            indicators[column] = 'Sell' if temp_df[['ma1']].iloc[-1].values < temp_df[['ma2']].iloc[
                -1].values else 'Keep'

        return pd.DataFrame(indicators, index=[0])

    def update_weights(self, portfolio):

        # Same portfolio and assets loaded but different weights => then calculate all variables
        self.portfolio = portfolio
        self.weights = np.array(list(map(lambda x: float(x), self.portfolio['market_value'].values))) / sum(
            list(map(lambda x: float(x), self.portfolio['market_value'].values)))
        self.value = np.round(sum(list(map(lambda x: float(x), self.portfolio['market_value'].values))), 2)
        self.volatility = np.round(self.get_annualized_volatility(), 2)
        self.returns = self.get_annualized_returns()
        self.sharpe_ratio = (self.returns - 0.02) / self.volatility

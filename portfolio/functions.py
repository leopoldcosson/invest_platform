import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf


def get_stocks_prices_from_alpaca(symbols: list, api_key: str, api_secret_key: str, interval='1Day'):
    """
    Get historical prices

    :param symbols: list of symbols to fetch
    :param api_key: api_key
    :param api_secret_key: api secret key
    :param interval: Interval => defined day per day but could be different to do intraday trading for instance
    :return: dict with a dataframe for every asset with date and close price
    """

    def fetch(url, next_page_token=None):

        if next_page_token:
            url += f'&page_token={next_page_token}'

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret_key
        }

        response = requests.get(url, headers=headers, verify=False)

        data = json.loads(response.text)
        try:
            next_page_token = 'stop' if str(data['next_page_token']) == 'None' else str(data['next_page_token'])
        except:
            next_page_token = 'stop'

        return data['bars'], next_page_token

    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={'%2C'.join(symbols)}&timeframe={interval}&start={start_date}&limit=10000&adjustment=raw&feed=iex"

    historical_prices = {}

    next_page_token = None
    while next_page_token != 'stop':
        data, next_page_token = fetch(url, next_page_token)
        historical_prices.update(data)

    prices = [[[dict['t'], dict['c']] for dict in historical_prices[symbol]] for symbol in symbols]

    stocks = {}
    for i in range(len(symbols)):
        try:
            stocks[symbols[i]] = pd.DataFrame(prices[i], columns=['date', 'close']).sort_values(by='date',
                                                                                                ascending=True)
        except:
            print('bug with: ', symbols[i])

    return stocks


def get_market_cap(symbols: list, api_key: str, api_secret_key: str, interval='1Day'):
    """
    Get market cap from our provider alpaca

    :param symbols: list of symbols to fetch
    :param api_key: api_key
    :param api_secret_key: api secret key
    :param interval: Interval => defined day per day but could be different to do intraday trading for instance
    :return: pd.DataFrame of symbol x market cap
    """

    def fetch(url, next_page_token=None):

        if next_page_token:
            url += f'&page_token={next_page_token}'

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret_key
        }

        response = requests.get(url, headers=headers, verify=False)

        data = json.loads(response.text)
        try:
            next_page_token = 'stop' if str(data['next_page_token']) == 'None' else str(data['next_page_token'])
        except:
            next_page_token = 'stop'

        return data['bars'], next_page_token

    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={'%2C'.join(symbols)}&timeframe={interval}&start={start_date}&limit=10000&adjustment=raw&feed=iex"

    historical_prices = {}

    next_page_token = None
    while next_page_token != 'stop':
        data, next_page_token = fetch(url, next_page_token)
        historical_prices.update(data)

    prices = [historical_prices[symbol][0]['c'] * historical_prices[symbol][0]['v'] for symbol in symbols]
    stocks = pd.DataFrame([prices], columns=symbols, index=['Market Cap'])

    return stocks


def get_symbols(api_key: str, api_secret_key: str) -> list:
    """
    Get all symbols available on alpaca

    :param api_key:
    :param api_secret_key:
    :return: list of symbols
    """

    url = "https://paper-api.alpaca.markets/v2/assets"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret_key
    }

    response = requests.get(url, headers=headers, verify=False)

    data = pd.DataFrame(json.loads(response.text))

    data = data.loc[(data['shortable'] == True)
                    & (data['tradable'] == True)
                    & (data['fractionable'] == True)
                    & (data['marginable'] == True)
                    & (data['exchange'] == 'NASDAQ')]

    symbols = list(data['symbol'].unique())

    return symbols


def get_stocks_prices_from_yahoo(symbols):
    portfolio_data = yf.download(' '.join(symbols), period="1y")['Close'].pct_change()
    return portfolio_data

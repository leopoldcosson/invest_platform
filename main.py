import random
import numpy as np
import time

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import urllib3
import openpyxl

from portfolio import *

urllib3.disable_warnings()

list_possible_symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BRK-B', 'TSLA', 'V', 'LLY', 'UNH', 'JPM',
                         'JNJ', 'XOM', 'WMT', 'MA', 'PG', 'AVGO', 'HD', 'ORCL', 'CVX', 'MRK', 'ABBV', 'KO', 'PEP',
                         'BAC', 'COST', 'ADBE', 'CSCO', 'TMO', 'MCD', 'CRM', 'PFE', 'NFLX', 'DHR', 'CMCSA', 'ABT',
                         'AMD', 'NKE', 'TMUS', 'WFC', 'DIS', 'UPS', 'TXN', 'PM', 'MS', 'INTC', 'CAT', 'BA', 'INTU',
                         'COP', 'UNP', 'AMGN', 'NEE', 'VZ', 'IBM', 'QCOM', 'LOW', 'BMY', 'DE', 'RTX', 'HON', 'AMAT',
                         'SPGI', 'GE', 'AXP', 'SCHW', 'BKNG', 'GS', 'SBUX', 'PLD', 'LMT', 'NOW', 'ELV', 'SYK', 'ISRG',
                         'BLK', 'ADP', 'T', 'MDLZ', 'GILD', 'TJX', 'CVS', 'ADI', 'MMC', 'LRCX', 'UBER', 'VRTX', 'ABNB',
                         'ZTS', 'C', 'CI', 'AMT', 'REGN', 'SLB', 'BDX', 'MO', 'FI', 'ITW']


def argmax(iterable) -> float:
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def highlight_sell(val) -> str:
    color = 'green' if val == 'Keep' else 'red'
    return f'background-color: {color}'


def optimisation_many_hikers(symbols_to_replace: list, symbols_to_keep: list):
    """
    Heuristic optimization algorithm to find one of the best solution. Kind of a multiple random walk with costs. Please see the medium article for more insights.

    :param symbols_to_replace: list of symbols to replace in our portfolio
    :param symbols_to_keep: list of symbols to keep in our portfolio
    :return: return the best portfolio and its sharpe ration
    """

    # First step : Downloading the 100th most traded asset of the year on the NASDAQ for the alpaca method or else load historical selection
    st.session_state['msg'] = st.toast('Downloading assets...', icon="🥞")
    time.sleep(1)
    if st.session_state['api_key'] != '':
        possible_symbols = get_symbols(st.session_state['api_key'], st.session_state['api_secret_key'])
        possible_symbols = list(
            get_market_cap(possible_symbols, st.session_state['api_key'], st.session_state['api_secret_key'],
                           interval='12Month').T.sort_values(by='Market Cap', ascending=False).iloc[:100].index)
    else:
        possible_symbols = list_possible_symbols.copy()

    # Second step : we only keep symbols that have a good momentum. Let's create a false portfolio with all possibles assets and get indicators to know which one to keep
    false_portfolio = st.session_state['portfolio'].portfolio.copy()
    for symbol in possible_symbols:
        if symbol not in list(false_portfolio.symbol.unique()):
            false_portfolio.loc[len(false_portfolio) + 1, :] = [symbol, 'NASDAQ', 0, 'long', 0.0]

    false_portfolio = Portfolio(st.session_state['api_key'], st.session_state['api_secret_key'],
                                false_portfolio)  # Build the fake portfolio dataframe
    indicators_df = false_portfolio.get_indicators()
    possible_symbols = list(
        indicators_df.T[indicators_df.T.iloc[:, 0] == 'Keep'].index)  # Keep only those with good momentum
    for elt in symbols_to_keep:
        if elt in possible_symbols:
            possible_symbols.remove(elt)

    # Third step : Calculate the initial amount
    initial_amount = st.session_state['portfolio'].value

    # Fourth step : We create random orders of asset
    random_orders = []
    nb_order = st.session_state['nb_iteration']
    for i in range(nb_order):
        temp_list = list(possible_symbols)
        random.shuffle(temp_list)
        random_orders.append(temp_list)

    # Fifth step : we explore each random order to find the first maximum local before reaching the total initial market value
    st.session_state['msg'] = st.toast('Testing every asset...', icon="🥞")
    initial_portfolio = false_portfolio.portfolio.loc[
                        false_portfolio.portfolio['symbol'].isin(possible_symbols + symbols_to_keep), :].copy()
    portfolio = Portfolio(st.session_state['api_key'], st.session_state['api_secret_key'], initial_portfolio)
    sharpe_ratios = []
    portfolios = []

    # Progress bar
    progress_text = "Testing different portfolios. Please wait."
    st.session_state['progress_bar'] = st.progress(0, text=progress_text)

    # Test of each order
    for i in range(nb_order):

        # Progress bar
        st.session_state['progress_bar'].progress(int(100 / nb_order * (i + 1)), text=progress_text)

        # For each random order, start from the initial portfolio
        portfolio.update_weights(initial_portfolio)
        portfolios.append([])
        sharpe_ratios.append([])

        # test symbol after symbol
        for symbol in random_orders[i]:

            if len(sharpe_ratios[i]) > 0:
                portfolio.update_weights(
                    portfolios[i][argmax(sharpe_ratios[i])])  # Update the portfolio with the best composition so far
            else:
                portfolio.update_weights(initial_portfolio)

            # Test 10 weights with the amount of market_value still free to be allocated
            possible_values = np.linspace(0, int(initial_amount - portfolio.value), num=10, dtype=int)
            if len(possible_values) == 10:
                for value in possible_values:
                    portfolio_df = portfolio.portfolio.copy()
                    portfolio_df.loc[portfolio_df['symbol'] == symbol, 'market_value'] = np.round(value, 2)
                    portfolio.update_weights(portfolio_df)
                    portfolios[i].append(portfolio.portfolio.copy())
                    sharpe_ratios[i].append(portfolio.sharpe_ratio)
            else:
                portfolios[i].extend([portfolio.portfolio.copy()]*10)
                sharpe_ratios[i].extend([portfolio.sharpe_ratio]*10)

    # 6th : We set to zero every sharpe ratio of portfolios having less than the minimum number of asset we want
    for i in range(len(portfolios)):
        for j in range(len(portfolios[i])):
            if len(portfolios[i][j].loc[portfolios[i][j]['market_value'].astype(float) > 0, :]) < len(symbols_to_keep) + \
                    st.session_state['nb_wanted_assets']:
                sharpe_ratios[i][j] = -10.0

    # 7th step : Keep only the best one
    st.session_state['msg'] = st.toast('Find the best one...', icon="🥞")
    max_sharpe_ratio = np.amax(np.array(sharpe_ratios))
    max_coordinates = np.array(sharpe_ratios).argmax()
    portfolios = np.array(portfolios)

    # Return the best portfolio and the max sharpe ratio of this portfolio
    return portfolios[max_coordinates // portfolios.shape[1]][max_coordinates % portfolios.shape[1]], max_sharpe_ratio


# Page configuration
st.set_page_config(
    page_title="Nara investment",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Introduction
st.title('💰 Investment platform')  # Title
st.divider()  # Grey horizontal bar
st.markdown(
    """
    👋 This platform has been developed by Leopold Cosson. Feel free to modify it to suit your needs but don't claim it as your own please.
    - You can see how this app has been built on my medium article here: https://medium.com/@leopold.cosson
    - The goal is to provide a place to handle your long-term investments and help you build your portfolio accordingly to basic financial knowledge.
"""
)
st.divider()

# Form to get the api_keys
with st.form(key='portfolioForm'):
    st.header('Portfolio')
    st.write('')

    # Load the Alpaca Portfolio
    st.subheader('From Alpaca Market:')
    col1, col2 = st.columns(2)
    st.session_state['api_key'] = col1.text_input(
        label='Api Key'
    )
    st.session_state['api_secret_key'] = col2.text_input(
        label='Api Secret Key'
    )

    # Load the Excel portfolio and format it as needed
    st.subheader('From Yahoo Finance')
    st.write('Load your portfolio from an excel file. Please respect the following format:')
    example_data = pd.DataFrame(
        [{'symbol': 'AAPL', 'market_value': 32.43}, {'symbol': 'AMZN', 'market_value': 2335.13}])
    col1, col2, col3 = st.columns(3)
    col2.dataframe(example_data)
    uploaded_file = st.file_uploader("Portfolio:", key='portfolioFile')
    if uploaded_file is not None:
        st.session_state['dataframe'] = pd.read_excel(uploaded_file)
        try:
            st.session_state['dataframe']['qty'] = 0
            st.session_state['dataframe']['side'] = 0
            st.session_state['dataframe']['exchange'] = 0
            st.session_state['dataframe'] = st.session_state['dataframe'][
                ['symbol', 'exchange', 'qty', 'side', 'market_value']]
        except:
            st.write('Please, respect the format above.')
    else:
        st.session_state['dataframe'] = None

    # Create our Portfolio instance
    col1, col2 = st.columns(2)
    if col1.form_submit_button(label='Load Portfolio', use_container_width=True, type='primary'):
        if st.session_state['dataframe'] is None:
            st.session_state['portfolio'] = Portfolio(st.session_state['api_key'], st.session_state['api_secret_key'])
        else:
            st.session_state['portfolio'] = Portfolio(st.session_state['api_key'], st.session_state['api_secret_key'],
                                                      st.session_state['dataframe'])

    # Reset the portfolio
    if col2.form_submit_button(label='Reset Portfolio', use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

# If our portfolio is loaded
if 'portfolio' in st.session_state:

    st.divider()
    with st.container():

        st.header('Your portfolio:')
        st.write('')

        # Repartition of assets
        fig_repartition = px.pie(st.session_state['portfolio'].portfolio, values='market_value', names='symbol')
        st.plotly_chart(fig_repartition, theme="streamlit", use_container_width=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Actual Value", '{:,}'.format(np.round(st.session_state['portfolio'].value, 2)) + ' $')
        col2.metric("Annualized Volatility", str(np.round(st.session_state['portfolio'].volatility * 100, 2)) + ' %')
        col3.metric("Sharpe Ratio", str(np.round(st.session_state['portfolio'].sharpe_ratio, 3)))
        st.caption('Risk Free Rate at 2%')

        css = '''
                [data-testid="metric-container"] {
                    width: fit-content;
                    margin: auto;
                }

                [data-testid="metric-container"] > div {
                    width: fit-content;
                    margin: auto;
                }

                [data-testid="metric-container"] label {
                    width: fit-content;
                    margin: auto;
                }
                '''
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

        # Details
        with st.expander('Details'):
            st.dataframe(st.session_state['portfolio'].portfolio, use_container_width=True, hide_index=True)

    # Cumulative returns
    st.divider()
    with st.container():
        st.header('Cumulative returns:')
        fig_returns = go.Figure()
        for column in st.session_state['portfolio'].historical_data.columns:
            fig_returns.add_trace(go.Scatter(
                x=st.session_state['portfolio'].historical_data.index,
                y=(st.session_state['portfolio'].historical_data[column] + 1).cumprod(),
                mode='lines',
                name=column
            ))
        st.plotly_chart(fig_returns, use_container_width=True)

    # Indicators
    with st.container():
        st.write('Based on the moving average 10 and 30. You should:')
        st.write('')

        indicators_df = st.session_state['portfolio'].get_indicators()
        symbols_to_replace = list(indicators_df.T[indicators_df.T.iloc[:, 0] == 'Sell'].index)
        symbols_to_keep = list(indicators_df.T[indicators_df.T.iloc[:, 0] == 'Keep'].index)
        st.dataframe(indicators_df.style.applymap(highlight_sell))

    # Optimization
    st.divider()
    with st.container():

        st.header('Optimization:')
        st.write(
            'We advice you to sell these red highlighted stocks and to let us calculate what shares you should buy to optimize your sharpe ratio.')
        st.write('')

        col1, col2 = st.columns(2)
        st.session_state['nb_iteration'] = col1.number_input('Number of iteration:', min_value=1, value=100)
        st.session_state['nb_wanted_assets'] = col2.number_input('How many asset minimum to replace the red ones ?',
                                                                 min_value=1, value=len(symbols_to_replace))

    if st.button('Start Optimization', use_container_width=True, type='secondary'):

        # Optimization process
        with st.spinner('Optimization in progress...'):

            # try:
            result_opti = optimisation_many_hikers(symbols_to_replace, symbols_to_keep)
            # except:
            #     result_opti = 'error. Try again'

        if result_opti != 'error. Try again':
            # Formatting results
            st.session_state['optimized_portfolio'] = pd.DataFrame(result_opti[0],
                                                                   columns=['symbol', 'exchange', 'qty', 'side',
                                                                            'market_value'])
            st.session_state['optimized_portfolio'] = st.session_state['optimized_portfolio'].loc[
                                                      st.session_state['optimized_portfolio']['market_value'].astype(
                                                          float) > 0.0, :]
            st.session_state['optimized_sharpe_ratio'] = result_opti[1]

            # Print results
            st.subheader('Result:')
            st.write(
                'According to our portfolio optimizer based on the historic simulated sharpe ratio, you should reorganize your portfolio as followed:')
            fig_opti = px.pie(st.session_state['optimized_portfolio'], values='market_value', names='symbol')
            st.plotly_chart(fig_opti, theme="streamlit", use_container_width=True)
            st.caption(f"Sharpe Ratio: {st.session_state['optimized_sharpe_ratio']}")

            st.dataframe(st.session_state['optimized_portfolio'][['symbol', 'market_value']],
                         hide_index=True)

        else:

            st.write(result_opti)

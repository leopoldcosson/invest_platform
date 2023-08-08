import streamlit as st
import random
import time
import plotly.express as px
import plotly.graph_objects as go
from portfolio import *
import urllib3
urllib3.disable_warnings()


def argmax(iterable)->float:
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def highlight_sell(val)->str:
    color = 'green' if val == 'Keep' else 'red'
    return f'background-color: {color}'


def optimisation_many_hikers(symbols_to_replace: list, symbols_to_keep: list):
    """
    Heuristic optimization algorithm to find one of the best solution. Kind of a multiple random walk with costs. Please see the medium article for more insights.

    :param symbols_to_replace: list of symbols to replace in our portfolio
    :param symbols_to_keep: list of symbols to keep in our portfolio
    :return: return the best portfolio and its sharpe ration
    """

    # First step : Downloading the 100th most traded asset of the year on the NASDAQ
    st.session_state['msg'] = st.toast('Downloading assets...', icon="🥞")
    time.sleep(1)
    possible_symbols = get_symbols(st.session_state['api_key'], st.session_state['api_secret_key'])
    possible_symbols = list(
        get_market_cap(possible_symbols, st.session_state['api_key'], st.session_state['api_secret_key'],
                       interval='12Month').T.sort_values(by='Market Cap', ascending=False).iloc[:100].index)

    # Second step : we only keep symbols that have a good momentum. Let's create a false portfolio with all possibles assets and get indicators to know which one to keep
    false_portfolio = st.session_state['portfolio'].portfolio.copy()
    for symbol in possible_symbols:
        if symbol not in list(false_portfolio.symbol.unique()):
            false_portfolio.loc[len(false_portfolio) + 1, :] = [symbol, 'NASDAQ', 0, 'long', 0.0]

    false_portfolio = Portfolio(st.session_state['api_key'], st.session_state['api_secret_key'], false_portfolio)  # Build the fake portfolio dataframe
    indicators_df = false_portfolio.get_indicators()
    possible_symbols = list(indicators_df.T[indicators_df.T.iloc[:, 0] == 'Keep'].index)  # Keep only those with good momentum
    for elt in symbols_to_keep:
        if elt in possible_symbols:
            possible_symbols.remove(elt)

    # Third step : Calculate the initial amount
    initial_amount = st.session_state['portfolio'].value

    # Fourth step : We create random orders of asset
    random_orders = []
    nb_order = 50
    for i in range(nb_order):
        temp_list = list(possible_symbols)
        random.shuffle(temp_list)
        random_orders.append(temp_list)

    # Fifth step : we explore each random order to find the first maximum local before reaching the total initial market value
    st.session_state['msg'] = st.toast('Testing every asset...', icon="🥞")
    initial_portfolio = false_portfolio.portfolio.loc[false_portfolio.portfolio['symbol'].isin(possible_symbols+symbols_to_keep), :].copy()
    portfolio = Portfolio(st.session_state['api_key'], st.session_state['api_secret_key'], initial_portfolio)
    sharpe_ratios = []
    portfolios = []

    # Progress bar
    progress_text = "Testing different portfolios. Please wait."
    st.session_state['progress_bar'] = st.progress(0, text=progress_text)

    # Test of each order
    for i in range(nb_order):

        # Progress bar
        st.session_state['progress_bar'].progress(int(100/nb_order*i)+1, text=progress_text)

        # For each random order, start from the initial portfolio
        portfolio.update_weights(initial_portfolio)
        portfolios.append([portfolio.portfolio])
        sharpe_ratios.append([portfolio.sharpe_ratio])

        # test symbol after symbol
        for symbol in random_orders[i]:

            portfolio.update_weights(portfolios[i][argmax(sharpe_ratios[i])])  # Update the portfolio with the best composition so far

            # Test 5 weights with the amount of market_value still free to be allocated
            possible_values = np.linspace(0.0, initial_amount-portfolio.value, num=5)
            for value in possible_values:
                portfolio_df = portfolio.portfolio.copy()
                portfolio_df.loc[portfolio_df['symbol'] == symbol, 'market_value'] = np.round(value, 2)
                portfolio.update_weights(portfolio_df)
                portfolios[i].append(portfolio.portfolio.copy())
                sharpe_ratios[i].append(portfolio.sharpe_ratio)

    # Sixth step : Keep only the best one
    st.session_state['msg'] = st.toast('Find the best one...', icon="🥞")
    max_sharpe_ratio = np.amax(np.array(sharpe_ratios))
    max_coordinates = np.array(sharpe_ratios).argmax()
    portfolios = np.array(portfolios)

    # Return the best portfolio and the max sharpe ratio of this portfolio
    return portfolios[max_coordinates//portfolios.shape[1]][max_coordinates%portfolios.shape[1]], max_sharpe_ratio


# Page configuration
st.set_page_config(
    page_title="Nara investment",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Introduction
st.title('Investment platform')
st.divider()
st.markdown(
    """
    👋 This platform has been developed by Leopold Cosson. Feel free to modify it to suit your needs but don't claim it as your own please.
    - You can see how this app has been built on my medium article here: https://medium.com/@leopold.cosson
    - The goal is to provide a place to handle your long-term investments and help you build your portfolio accordingly to basic financial knowledge.
"""
)
st.divider()

#Form to get the api_keys
with st.form('api_keys'):
    st.header('Api keys from Alpaca')
    st.write('')
    col1, col2 = st.columns(2)
    st.session_state['api_key'] = col1.text_input(
        label='Api Key'
    )
    st.session_state['api_secret_key'] = col2.text_input(
        label='Api Secret Key'
    )

    # Create our Portfolio instance
    if st.form_submit_button(label='Load Portfolio', use_container_width=True, type='primary'):
        st.session_state['portfolio'] = Portfolio(st.session_state['api_key'], st.session_state['api_secret_key'])

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
        col1.metric("Actual Value", '{:,}'.format(st.session_state['portfolio'].value) + ' $')
        col2.metric("Annualized Volatility", str(st.session_state['portfolio'].volatility) + ' %')
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
            st.dataframe(st.session_state['portfolio'].portfolio, use_container_width=True)

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
        st.dataframe(indicators_df.style.applymap(highlight_sell), use_container_width=True)

    # Optimization
    st.divider()
    with st.container():

        st.header('Optimization:')
        st.write('We advice you to sell these red highlighted stocks and to let us calculate what shares you should buy to optimize your sharpe ratio.')
        st.write('')

    if st.button('Start Optimization', use_container_width=True, type='secondary'):

        # Optimization process
        with st.spinner('Optimization in progress...'):

            result_opti = optimisation_many_hikers(symbols_to_replace, symbols_to_keep)

            # Formatting results
            st.session_state['optimized_portfolio'] = pd.DataFrame(result_opti[0], columns=['symbol', 'exchange', 'qty', 'side', 'market_value'])
            st.session_state['optimized_portfolio'] = st.session_state['optimized_portfolio'].loc[st.session_state['optimized_portfolio']['market_value'].astype(float) > 0.0, :]
            st.session_state['optimized_sharpe_ratio'] = result_opti[1]

        # Print results
        st.subheader('Result:')
        st.write('According to our portfolio optimizer based on the historic simulated sharpe ratio, you should reorganize your portfolio as followed:')
        fig_opti = px.pie(st.session_state['optimized_portfolio'], values='market_value', names='symbol')
        st.plotly_chart(fig_opti, theme="streamlit", use_container_width=True)
        st.caption(f"Sharpe Ratio: {st.session_state['optimized_sharpe_ratio']}")

        st.dataframe(st.session_state['optimized_portfolio'][['symbol', 'exchange', 'market_value']])

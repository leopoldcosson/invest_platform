# Investment Platform

Nara Investment is a python solution working with streamlit to optimize your long term investment based on maximizing the sharpe ratio. It uses the data provider and broker Alpaca Market for now. (I'm looking for another data provider, so we can just upload an excel of a portfolio.)

### Alpaca Market

Alpaca markets is a broker and data provider for algorithmic trading quite easy to use and very useful. Feel free to open an account and get your keys. Otherwise you won't be able to use the platform.

### Optimization algorithm

Our solution provide a heuristic optimizer based on a multiple random walk with costs. Please see the medium article I have written about this platform to better understand how it works.

## Authors

- [@leopoldcosson](https://www.github.com/leopoldcosson)


## Deployment

To deploy this project :

1. Install streamlit

```bash
  pip install streamlit
```

2. Open the right path

```bash
  cd .../invest_platform/
```

3. And then run

```bash
  streamlit run main.py
```


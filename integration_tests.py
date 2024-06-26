import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PortOpt import get_stock_data, plot_efficient_frontier, simulate_portfolio_performance

class TestPortfolioFunctions(unittest.TestCase):

    @patch('yfinance.download')
    def test_get_stock_data(self, mock_yf_download):
        # Mock the yfinance.download function
        mock_yf_download.return_value = pd.DataFrame({
            'Adj Close': {
                'AAPL': [150, 151, 152],
                'GOOGL': [2800, 2810, 2820]
            }
        })
        
        tickers = ['AAPL', 'GOOGL']
        start_date = '2023-01-01'
        end_date = '2023-01-03'
        
        result = get_stock_data(tickers, start_date, end_date)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 2))
        mock_yf_download.assert_called_once_with(tickers, start=start_date, end=end_date)

    @patch('matplotlib.pyplot.show')
    def test_plot_efficient_frontier(self, mock_show):
        returns = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
        optimal_weights = np.array([0.3, 0.3, 0.4])
        
        plot_efficient_frontier(returns, optimal_weights)
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_simulate_portfolio_performance(self, mock_show):
        ave_return = 0.1
        ave_std = 0.2
        num_years = 10
        initial_investment = 1000
        
        simulate_portfolio_performance(ave_return, ave_std, num_years, initial_investment)
        
        mock_show.assert_called_once()

import unittest
import numpy as np
import pandas as pd
from PortOpt import (
    calculate_returns,
    calculate_portfolio_variance,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    optimize_portfolio
)

class TestPortfolioFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.prices = pd.DataFrame({
            'A': [100, 102, 104, 103, 105],
            'B': [50, 51, 52, 51, 53]
        })
        self.returns = calculate_returns(self.prices)
        self.weights = np.array([0.6, 0.4])
        self.cov_matrix = self.returns.cov()

    def test_calculate_returns(self):
        expected_returns = pd.DataFrame({
            'A': [np.nan, 0.02, 0.019608, -0.009615, 0.019417],
            'B': [np.nan, 0.02, 0.019608, -0.019231, 0.039216]
        })
        pd.testing.assert_frame_equal(self.returns, expected_returns, check_less_precise=True)

    def test_calculate_portfolio_variance(self):
        variance = calculate_portfolio_variance(self.weights, self.cov_matrix)
        self.assertAlmostEqual(variance, 0.00015, places=5)

    def test_calculate_portfolio_return(self):
        portfolio_return = calculate_portfolio_return(self.weights, self.returns)
        self.assertAlmostEqual(portfolio_return, 0.0714, places=4)

    def test_calculate_portfolio_volatility(self):
        volatility = calculate_portfolio_volatility(self.weights, self.cov_matrix)
        self.assertAlmostEqual(volatility, 0.1936, places=4)

    def test_optimize_portfolio(self):
        optimal_weights = optimize_portfolio(self.returns)
        self.assertEqual(len(optimal_weights), 2)
        self.assertAlmostEqual(np.sum(optimal_weights), 1, places=6)
        self.assertTrue(all(0 <= w <= 1 for w in optimal_weights))

if __name__ == '__main__':
    unittest.main()
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
        pd.testing.assert_frame_equal(self.returns, expected_returns, atol=1e-6, rtol=1e-6)

    def test_calculate_portfolio_variance(self):
        expected_variance = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))
        variance = calculate_portfolio_variance(self.weights, self.cov_matrix)
        self.assertAlmostEqual(variance, expected_variance, places=6)

    def test_calculate_portfolio_return(self):
        expected_return = np.sum(self.returns.mean() * self.weights) * 252
        portfolio_return = calculate_portfolio_return(self.weights, self.returns)
        self.assertAlmostEqual(portfolio_return, expected_return, places=4)

    def test_calculate_portfolio_volatility(self):
        variance = calculate_portfolio_variance(self.weights, self.cov_matrix)
        expected_volatility = np.sqrt(variance) * np.sqrt(252)
        volatility = calculate_portfolio_volatility(self.weights, self.cov_matrix)
        self.assertAlmostEqual(volatility, expected_volatility, places=4)

    def test_optimize_portfolio(self):
        optimal_result = optimize_portfolio(self.returns)
        self.assertEqual(len(optimal_result.x), 2)
        self.assertAlmostEqual(np.sum(optimal_result.x), 1, places=6)
        self.assertTrue(all(0 <= w <= 1 for w in optimal_result.x))

if __name__ == '__main__':
    unittest.main()
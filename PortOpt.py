"""
This file optimizes and simulates the performance of a stock portfolio using historical data from Yahoo Finance.
It calculates optimal stock weights, plots the efficient frontier, and simulates future portfolio performance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import itertools


def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_returns(data):
    return data.pct_change()

def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_portfolio_return(weights, returns):
    # Annualized portfolio return
    return np.sum(returns.mean() * weights) * 252

def calculate_portfolio_volatility(weights, cov_matrix):
    # Annualized portfolio volatility; uses sqrt(252) instead of 252 to annualize
    return np.sqrt(calculate_portfolio_variance(weights, cov_matrix)) * np.sqrt(252) 

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    portfolio_return = calculate_portfolio_return(weights, returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, returns.cov())
    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe  # Negated for minimization

def optimize_portfolio(returns, risk_free_rate=0.02):
    """
    Optimize the portfolio weights to maximize the Sharpe ratio.

    Args:
        returns (pandas.DataFrame): Daily returns of the stocks.
        risk_free_rate (float, optional): Risk-free rate of return. Default is 0.02.

    Returns:
        scipy.optimize.OptimizeResult: Optimization result containing the optimal weights.
    """
    num_assets = len(returns.columns)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Ensure weights sum to 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # Weights between 0 and 1
    initial_weights = num_assets * [1. / num_assets]  # Equal weight start

    optimal_result = minimize(negative_sharpe_ratio, initial_weights, args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal_result

def plot_efficient_frontier(returns, optimal_result, risk_free_rate, width, height):
    num_portfolios = 500  # number of random portfolios generated
    num_assets = len(returns.columns)
    
    # Create a grid of weights
    step = 0.1
    weights = list(itertools.product(np.arange(0, 1.01, step), repeat=num_assets))
    weights = [w for w in weights if np.isclose(sum(w), 1)]
    
    # Limit to around 250 portfolios
    if len(weights) > num_portfolios:
        weights = weights[:num_portfolios]
    
    results = np.zeros((3, len(weights)))
    for i, w in enumerate(weights):
        w = np.array(w)  # Convert tuple to numpy array
        portfolio_return = calculate_portfolio_return(w, returns)
        portfolio_volatility = calculate_portfolio_volatility(w, returns.cov())
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio

    optimal_weights = optimal_result.x
    optimal_return = calculate_portfolio_return(optimal_weights, returns)
    optimal_volatility = calculate_portfolio_volatility(optimal_weights, returns.cov())
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

    # Calculate individual stock returns and volatilities
    stock_returns = returns.mean() * 252
    stock_volatilities = returns.std() * np.sqrt(252)
    stock_sharpe_ratios = (stock_returns - risk_free_rate) / stock_volatilities

    dpi = 100  # Adjust this value if needed
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Calculate proportional sizes
    point_size = width * height / 40000
    font_size_small = min(width, height) / 100
    font_size_medium = min(width, height) / 80
    font_size_large = min(width, height) / 60

    # Plot efficient frontier
    scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', s=point_size)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio', fontsize=font_size_medium)
    cbar.ax.tick_params(labelsize=font_size_small)
    
    # Plot individual stocks
    for ticker, return_, volatility in zip(returns.columns, stock_returns, stock_volatilities):
        ax.scatter(volatility, return_, color='red', s=point_size*2)
        ax.annotate(ticker, (volatility, return_), xytext=(5, 5), textcoords='offset points', fontsize=font_size_small)
    
    # Plot optimal portfolio
    ax.scatter(optimal_volatility, optimal_return, color='green', marker='*', s=point_size*4, label='Optimal Portfolio')
    
    ax.set_xlabel('Volatility', fontsize=font_size_medium)
    ax.set_ylabel('Return', fontsize=font_size_medium)
    ax.set_title('Efficient Frontier', fontsize=font_size_large)
    ax.legend(fontsize=font_size_small)
    ax.tick_params(axis='both', which='major', labelsize=font_size_small)
    
    plt.tight_layout()

    print("Stocks and their optimal weights:")
    for stock, weight in zip(returns.columns, optimal_weights):
        print(f"{stock}: {weight:.4f}")
    print("\nOptimal Portfolio:")
    print(f"Sharpe Ratio: {optimal_sharpe:.4f}")
    print(f"Volatility: {optimal_volatility:.4f}")
    print(f"Return: {optimal_return:.4f}")

    return fig, optimal_return, optimal_volatility, optimal_sharpe


def simulate_portfolio_performance(ave_return, ave_std, num_years, initial_investment=1000):
    """
    Simulate the performance of a portfolio over a specified number of years.

    Args:
        ave_return (float): Average annual return of the portfolio.
        ave_std (float): Standard deviation of the portfolio's annual returns.
        num_years (int): Number of years to simulate.
        initial_investment (float, optional): Initial investment amount. Default is 1000.
    """
    total_return = initial_investment
    returns = []

    for i in range(num_years):
        annual_return = np.random.normal(ave_return, ave_std)  # Random return based on mean and std
        total_return *= (1 + annual_return)
        returns.append(total_return)

    years = range(1, num_years + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(years, returns)
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value')
    plt.title('Simulated Portfolio Performance')
    plt.grid(True)
    plt.show()

    print(f"Initial Investment: ${initial_investment:.2f}")
    print(f"Final Portfolio Value: ${total_return:.2f}")

def main():
    tickers = ['o', 'cost', 'aapl', 'gs']
    start_date = '2015-01-01'
    end_date = '2024-04-08'
    num_years = 10
    initial_investment = 1000
    risk_free_rate = 0.02

    stock_data = get_stock_data(tickers, start_date, end_date)
    returns = calculate_returns(stock_data)
    optimal_result = optimize_portfolio(returns, risk_free_rate)
    optimal_return, optimal_volatility, optimal_sharpe = plot_efficient_frontier(returns, optimal_result, risk_free_rate)
    # simulate_portfolio_performance(optimal_return, optimal_volatility, num_years, initial_investment)

if __name__ == "__main__":
    main()
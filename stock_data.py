import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_stock_data(tickers, start_date, end_date):
    # Retrieve stock data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_returns(data):
    # Calculate daily returns of the stock prices
    returns = data.pct_change()
    return returns

def calculate_portfolio_variance(weights, cov_matrix):
    # Calculate the variance of a portfolio
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_portfolio_return(weights, returns):
    # Calculate the annualized return of a portfolio
    return np.sum(returns.mean() * weights) * 252

def calculate_portfolio_volatility(weights, cov_matrix):
    # Calculate the annualized volatility of a portfolio
    return np.sqrt(calculate_portfolio_variance(weights, cov_matrix)) * np.sqrt(252)

def optimize_portfolio(returns):
    # Perform portfolio optimization
    num_assets = len(returns.columns)
    args = (returns.cov(),)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]
    optimal_weights = minimize(calculate_portfolio_volatility, initial_weights, args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal_weights.x

def plot_efficient_frontier(returns, optimal_weights):
    # Generate random portfolios and plot the efficient frontier
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return = calculate_portfolio_return(weights, returns)
        portfolio_volatility = calculate_portfolio_volatility(weights, returns.cov())
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = (portfolio_return - 0.02) / portfolio_volatility
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
    max_sharpe_allocation = weights_record[max_sharpe_idx]
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.plot(sdp, rp, 'r*', markersize=15)
    plt.grid(True)
    plt.show()
    
    print("Optimal Weights:", optimal_weights)
    print("Sharpe Ratio:", results[2, max_sharpe_idx])
    print("Volatility:", results[1, max_sharpe_idx])
    print("Return:", results[0, max_sharpe_idx])
    return results[:,max_sharpe_idx] # returns average return, average volatility, and sharpe ratio of the best portfolio found

def simulate_portfolio_performance(ave_return, ave_std, num_years, initial_investment=1000):
    total_return = initial_investment
    returns = []
    
    for i in range(num_years):
        # Calculate a random return based on ave_return and ave_std
        annual_return = np.random.normal(ave_return, ave_std)
        total_return *= (1 + annual_return)
        returns.append(total_return)
    
    # Plot the simulated portfolio performance
    years = range(1, num_years + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(years, returns)
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value')
    plt.title('Simulated Portfolio Performance')
    plt.grid(True)
    plt.show()
    
    # Print the final portfolio value
    print(f"Initial Investment: ${initial_investment:.2f}")
    print(f"Final Portfolio Value: ${total_return:.2f}")

# Example usage
tickers = ['msft', 'bac', 'xom', 'tsla']
start_date = '2011-01-01'
end_date = '2024-04-08'
num_years = 10
initial_investment = 1000

stock_data = get_stock_data(tickers, start_date, end_date)
returns = calculate_returns(stock_data)
optimal_weights = optimize_portfolio(returns)
results = plot_efficient_frontier(returns, optimal_weights)
#print(results[1])
#print(results[0])
simulate_portfolio_performance(results[0], results[1], num_years, initial_investment)
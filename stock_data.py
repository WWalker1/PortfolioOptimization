import pandas as pd 
import yfinance as yf
from datetime import datetime, timedelta


# Retrieve stock data from Yahoo Finance
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']  


def get_stock_returns(stock_data): 
    return stock_data.pct_change()


def create_cov_matrix(stock_returns):
    cov_matrix = stock_returns.cov()
    return cov_matrix


def main():
    # Define a list of stock tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    # Set date range (e.g., last 1 year)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Get stock data
    stock_data = get_stock_data(tickers, start_date, end_date)
    
    # Calculate returns
    stock_returns = get_stock_returns(stock_data)
    
    # Create covariance matrix
    cov_matrix = create_cov_matrix(stock_returns)
    
    # Print results
    print("Stock Data:")
    print(stock_data.head())
    print("\nStock Returns:")
    print(stock_returns.head())
    print("\nCovariance Matrix:")
    print(cov_matrix)

if __name__ == "__main__":
    main()



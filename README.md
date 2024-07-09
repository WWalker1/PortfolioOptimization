# Portfolio Optimizer

A desktop application for optimizing stock portfolios using modern portfolio theory.

## Screenshots

![image](https://github.com/WWalker1/PortfolioOpt-Website/assets/70979927/68f71a6f-475b-4b76-afec-99d4437b1370)

![Screenshot 2024-07-09 182358](https://github.com/WWalker1/PortfolioOpt-Website/assets/70979927/7b53bd45-cb47-4c33-afea-9b45444cfc12)

## Features

- Full-screen application with a modern, intuitive interface
- Input multiple stock tickers for portfolio optimization
- Set custom date ranges for historical data analysis
- Adjustable risk-free rate
- Visualize the efficient frontier within the application
- View optimal portfolio weights and key metrics

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/portfolio-optimizer.git
   ```
2. Navigate to the project directory:
   ```
   cd portfolio-optimizer
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python PortfolioOptimizerApp.py
   ```
2. In the main menu:
   - Enter stock tickers separated by commas (e.g., AAPL, GOOGL, MSFT)
   - Set the start and end dates for historical data
   - Enter the risk-free rate (default is 0.02)
   - Click "Optimize Portfolio"
3. View the efficient frontier plot and optimization results
4. Click "Back to Main Screen" to return to the input screen

## Dependencies

- Python 3.7+
- PyQt5
- Matplotlib
- NumPy
- Pandas
- yfinance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

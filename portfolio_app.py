import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QDateEdit, QStackedWidget, 
                             QFrame, QGridLayout, QSplitter)
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QScreen, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import PortOpt

class ModernButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                font-size: 65px;
                margin: 4px 2px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

class PortfolioOptimizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portfolio Optimizer")
        self.setWindowState(Qt.WindowFullScreen)
        
        # Get screen size
        screen = QApplication.primaryScreen()
        size = screen.size()
        self.width = size.width()
        self.height = size.height()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.setup_input_screen()
        self.setup_plot_screen()

        # Set modern style
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                color: #333333;
                font-family: Arial, sans-serif;
            }
            QLabel {
                font-size: 65px;
                margin-bottom: 5px;
            }
            QLineEdit, QDateEdit {
                font-size: 60px;
                padding: 10px;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QTextEdit {
                font-size: 60px;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
        """)

    def setup_input_screen(self):
        input_widget = QWidget()
        input_layout = QGridLayout(input_widget)

        # Title
        title_label = QLabel("Portfolio Optimizer")
        title_label.setStyleSheet("font-size: 120px; font-weight: bold; margin-bottom: 20px;")
        input_layout.addWidget(title_label, 0, 0, 1, 2, Qt.AlignCenter)

        # Left column: Inputs
        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.StyledPanel)
        input_frame_layout = QVBoxLayout(input_frame)

        # Stock input
        input_frame_layout.addWidget(QLabel("Enter stock tickers (comma-separated):"))
        self.stock_input = QLineEdit()
        self.stock_input.setFixedHeight(int(self.height * 0.05))
        input_frame_layout.addWidget(self.stock_input)

        # Date inputs
        date_layout = QGridLayout()
        self.start_date = QDateEdit()
        self.start_date.setFixedHeight(int(self.height * 0.05))
        self.start_date.setDate(QDate(2015, 1, 1))
        self.end_date = QDateEdit()
        self.end_date.setFixedHeight(int(self.height * 0.05))
        self.end_date.setDate(QDate.currentDate())

        date_layout.addWidget(QLabel("Start Date:"), 0, 0)
        date_layout.addWidget(self.start_date, 0, 1)
        date_layout.addWidget(QLabel("End Date:"), 1, 0)
        date_layout.addWidget(self.end_date, 1, 1)
        input_frame_layout.addLayout(date_layout)

        # Risk-free rate input
        input_frame_layout.addWidget(QLabel("Risk-free Rate:"))
        self.risk_free_rate = QLineEdit()
        self.risk_free_rate.setFixedHeight(int(self.height * 0.05))
        self.risk_free_rate.setText("0.02")
        input_frame_layout.addWidget(self.risk_free_rate)

        # Optimize button
        self.optimize_button = ModernButton("Optimize Portfolio")
        self.optimize_button.setFixedHeight(int(self.height * 0.07))
        self.optimize_button.clicked.connect(self.optimize_portfolio)
        input_frame_layout.addWidget(self.optimize_button)

        input_layout.addWidget(input_frame, 1, 0)

        # Right column: Results and Information
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.StyledPanel)
        results_frame_layout = QVBoxLayout(results_frame)

        results_frame_layout.addWidget(QLabel("Optimization Results:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_frame_layout.addWidget(self.results_text)

        # Add some information about the app
        info_text = QLabel("This application optimizes stock portfolios using modern portfolio theory. "
                           "Enter your desired stocks, set the date range, and click 'Optimize Portfolio' "
                           "to see the results.")
        info_text.setWordWrap(True)
        info_text.setStyleSheet("font-size: 55px; margin-top: 20px;")
        results_frame_layout.addWidget(info_text)

        input_layout.addWidget(results_frame, 1, 1)

        self.stacked_widget.addWidget(input_widget)

    def setup_plot_screen(self):
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Matplotlib canvas
        self.plot_canvas = FigureCanvas(Figure(figsize=(self.width/100, self.height/100)))
        plot_layout.addWidget(self.plot_canvas)

        # Back button
        self.back_button = ModernButton("Back to Main Screen")
        self.back_button.setFixedHeight(int(self.height * 0.07))
        self.back_button.clicked.connect(self.show_input_screen)
        plot_layout.addWidget(self.back_button)

        self.stacked_widget.addWidget(plot_widget)

    def optimize_portfolio(self):
        tickers = [ticker.strip() for ticker in self.stock_input.text().split(',')]
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        risk_free_rate = float(self.risk_free_rate.text())

        # Get stock data and calculate returns
        stock_data = PortOpt.get_stock_data(tickers, start_date, end_date)
        returns = PortOpt.calculate_returns(stock_data)

        # Optimize portfolio
        optimal_result = PortOpt.optimize_portfolio(returns, risk_free_rate)

        # Plot efficient frontier
        button_height = int(self.height * 0.07)
        fig, optimal_return, optimal_volatility, optimal_sharpe = PortOpt.plot_efficient_frontier(returns, optimal_result, risk_free_rate, self.width, self.height - button_height)

        # Display plot in the application
        self.plot_canvas.figure = fig
        self.plot_canvas.draw()

        # Display results
        self.display_results(optimal_result, returns, risk_free_rate, optimal_return, optimal_volatility, optimal_sharpe)

        # Show plot screen
        self.stacked_widget.setCurrentIndex(1)

    def display_results(self, optimal_result, returns, risk_free_rate, optimal_return, optimal_volatility, optimal_sharpe):
        optimal_weights = optimal_result.x

        results = "Optimal Portfolio:\n\n"
        results += "Stocks and their optimal weights:\n"
        for stock, weight in zip(returns.columns, optimal_weights):
            results += f"{stock}: {weight:.4f}\n"
        results += f"\nSharpe Ratio: {optimal_sharpe:.4f}\n"
        results += f"Volatility: {optimal_volatility:.4f}\n"
        results += f"Return: {optimal_return:.4f}\n"

        self.results_text.setText(results)

    def show_input_screen(self):
        self.stacked_widget.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PortfolioOptimizerApp()
    window.show()
    sys.exit(app.exec_())
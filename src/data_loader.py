# data_loader.py
"""
Module for upload finance data
"""
import numpy as np

# Patch NumPy for scikit-cuda compatibility
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'complex'):
    np.complex = np.complex128

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataLoader:
    """Class for loading and processing financial data"""

    def __init__(self):
        self.data = None

    def download_stock_data(self, ticker='AAPL', period='2y'):
        """
        Loading stock data from Yahoo Finance
        Parameters:
        ------------
        ticker: str
            Ticker stock data (for example: 'AAPL', 'GOOGL', 'TSLA')
        period: str 
            data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        Returns:
        ------------
        pandas.DataFrame
            Data with stock price
        """
        print(f"Loading data for {ticker} for the period {period}...")

        try:
            stock = yf.Ticker(ticker)
            self.data = stock.history(period=period)

            if self.data.empty:
                raise ValueError(f"Not data for ticker {ticker}")

            print(f"Uploaded {len(self.data)} traiding days")
            print(f"Period: {self.data.index[0].date()} to {self.data.index[-1].date()} ")
            print(f"Current price: ${self.data['Close'].iloc[-1]:.2f}")

            return self.data

        except Exception as e:
            print(f"Error loading: {e}")
            return None

    def calculate_returns(self):
        """
        Calculation of daily yields
        Returns:
        numpy.ndarray
            Array of daily returns
        """
        if self.data is None:
            raise ValueError("At first upload data through download_stock_data()")

        #Daily yields = (Price_today - Price_yesterday) / Price_yesterday
        returns = self.data['Close'].pct_change().dropna()

        print(f"\n Return statistics:")
        print(f"  Average daily return: {returns.mean()*100:.4f}%")
        print(f"  Volatility (std): {returns.std()*100:.4f}%")
        print(f"  Minimum: {returns.min()*100:.2f}%")
        print(f"  Maximum: {returns.max()*100:.2f}%")

        return returns.values

    def get_statistics(self):
        """
        Get statistics paramethers for simulation

        Returns:
        ------------
        dict
            Dictionary with parameters: mu (profitability), sigma (Volatility), S0 (current price)
        ------------
        """
        if self.data is None:
            raise ValueError("Upload data")

        returns = self.calculate_returns()

        # Annualised yield (252 trading days per year)
        mu = returns.mean()

        # Annualised volatility
        sigma = returns.std() * np.sqrt(252)

        # Current price
        S0 = self.data['Close'].iloc[-1]

        params = {
            'mu': mu, # expected annualised yield
            'sigma': sigma, # Annualised volatility
            'S0': S0, # Current price
            'ticker': None, # Save ticker for reports
        }
        
        print(f"\n Parameters for monteCarlo simulation:")
        print(f" m (mu - return) : {mu * 100:.2f}% per annum")
        print(f"  q (sigma - volatility):{sigma*100:.2f}% per annum")
        print(f" S0 (current price): ${S0:.2f}")

        return params

    def save_data(self, filename = 'stock_data.csv'):
        """Save data in CSV"""
        if self.data is None:
            raise ValueError("No data for save")

        filepath = f"data/raw/{filename}"
        self.data.to_csv(filepath)
        print(f"Data saved in {filepath}")

# Example
if __name__ == "__main__":
    #create loader
    loader = DataLoader()

    # Upload Apple data for 2 years
    data = loader.download_stock_data(ticker='AAPL', period='2y')

    # Get parameters for simulation
    params = loader.get_statistics()

    # Save data
    loader.save_data('AAPF:2y.csv')
              
        
    
    
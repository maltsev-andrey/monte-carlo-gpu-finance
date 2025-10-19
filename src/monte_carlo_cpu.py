# monte_carlo_cpu.py
"""
CPU version for Monte Varlo simulation
Basic algoritm
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

import time
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SimulationParams:
    """Simulation Parameters"""
    S0: float  # Initial stock price
    K: float   # Option strike price
    T: float   # Time to expiration (yrs)
    r: float   # Risk free interest time
    sigma: float          # Volatility
    num_simulations: int  # Number of simulations
    num_steps: int        # Numbers of time steps

class MonteCarloEngine:
    """the engine for monte Carlo simulations on CPU"""
    def __init__(self, params: SimulationParams):
        self.params = params
        self.results = None

    def simulate_stock_prices(self) -> np.ndarray:
        """
        Simulations of stock price trajectories

        Returns:
        ------------
        np.ndarray, shape (num_simulations, num_steps)
                Matrix of the prices simulation
        """

        print(f"Start CPU simulation....")
        print(f"  Simulations: {self.params.num_simulations:,}")
        print(f"  Steps: {self.params.num_steps}")

        start_time = time.time()

        # Parameters
        S0 = self.params.S0
        T = self.params.T
        r = self.params.r
        sigma = self.params.sigma
        N = self.params.num_simulations
        M = self.params.num_steps

        # Temporary step
        dt = T / M

        # pre-calculations of coefficients
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        # random number generation (N simulations x M steps)
        random_numbers = np.random.standard_normal((N,M))

        # Initializing the price array
        prices = np.zeros((N,M + 1))
        prices[:, 0] = S0 #starting price

        # Trajectories simulation
        for i in range(M):
            prices[:, i + 1] = prices[:, i] * np.exp(drift + vol * random_numbers[:, i])

        elapsed = time.time() - start_time
        print(f" Simulation complited in {elapsed:.4f} sec.")

        self.results = prices
        return prices

    def price_european_call(self, K: float = None) -> Tuple[float, float]:
        """
        Evaluation EU Call Option

        Call Option = right to buy a share at fixed price "K"
        Payoff = max(S_T - K, 0)

        Parameters:
        ----------
        K: float
            Straike price (if "None" is uses as params)

        Returns:
        ----------
        tuple (price, std_error)
            Option price and standard error
        """
        if self.results is None:
            self.simulate_stock_prices()
        
        if K is None:
            K = self.params.K

        # Final price (last column)
        final_prices = self.results[:, -1]

        #Payoff = max(S_T - K, 0)
        payoffs = np.maximum(final_prices - K, 0)

        # Discounting to the current moment
        discount_factor = np.exp(-self.params.r * self.params.T)        
        discounted_payoffs = payoffs * discount_factor

        # Option price = average of discounted payoffs
        option_price = np.mean(discounted_payoffs)

        # Standard error
        std_error = np.std(discounted_payoffs) / np.sqrt(self.params.num_simulations)

        return option_price, std_error

    def price_european_put(self, K: float = None) -> Tuple[float, float]:
        """
        Valuation of European "Put" option
        Put option = right to sell actions by fixing price "K"
        Payoff = max(K-S_T, 0)
        """

        if self.results is None:
            self.simulate_stock_prices()

        if K is None:
            K = self.params.K

        final_prices = self.results[:, -1]
        payoffs = np.maximum(K - final_prices, 0)

        discount_factor = np.exp(-self.params.r * self.params.T)
        discounted_payoffs = payoffs * discount_factor

        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.params.num_simulations)

        return option_price, std_error

    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        VaR = max. loss with a given propability
        
        Parameters:
        -----------
        confidence_level : float
            Trust level (for example: 0.95 = 95%)
            
        Returns:
        --------
        float
            VaR value (positive number = loss)
        """

        if self.results is None:
            self.simulate_stock_prices()

        # Final prices
        final_prices = self.results[:, -1]

        # Profit/loss repated to the starting price
        returns = (final_prices - self.params.S0) / self.params.S0

        # VaR = quantile lf loss distribution
        var = -np.percentile(returns, (1 - confidence_level) * 100)

        return var

    def get_statistics(self) -> dict:
        """
        Getting Simulation Statistics
        
        Returns:
        --------
        dict
            dictionary with simulations
        """        
        if self.results is None:
            self.simulate_stock_prices()

        final_prices = self.results[:, -1]

        stats = {
            'mean_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_price': np.std(final_prices),
            'min_price': np.min(final_prices),
            'max_price': np.max(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_95': np.percentile(final_prices, 95)
        }

        return stats

# usage example        
if __name__ == "__main__":
    print("=" * 60)
    print("monte Carlo Simulation - CPU Version")
    print("=" * 60)

    # Simulation parameters
    # Example: a stock costs $100, we want to price an option with a strike price of $105
    # 1 year ahead
    params = SimulationParams(
        S0 = 100.0,  # Current price
        K = 105.0,   # Option strike
        T = 1.0,     # 1 year
        r = 0.05,    # 5% risk-free rate (can be taken from US Treasury)
        sigma = 0.25,              # real volontility
        num_simulations = 1_000_000, # 1 mil simulation
        num_steps =252             # 252  traid days 
    )

    # Create engine
    engine = MonteCarloEngine(params)

    # Start simulation
    prices = engine.simulate_stock_prices()

    print("\n" + "=" * 60)
    print("Results Simulation")
    print("=" * 60)

    # Price statistic
    stats = engine.get_statistics()
    print(f"\n Final price statisctics:")
    print(f"   Mean price: ${stats['mean_price']:.2f}")
    print(f"   Median: ${stats['median_price']:.2f}")
    print(f"   Std. deviation: ${stats['std_price']:.2f}")
    print(f"   Range: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}")
    print(f"   5-95 percentile: ${stats['percentile_5']:.2f} - ${stats['percentile_95']:.2f}")   

    # Evaluate Call options
    call_price, call_se = engine.price_european_call()
    print(f"\n Call option (rights to buy at ${params.K}):")
    print(f"   Price: ${call_price:.4f} ± ${call_se:.4f}")

    # Price "Put" option
    put_price, put_se = engine.price_european_put()
    print(f"\n Put option (right to sell at ${params.K}):")
    print(f"   Price: ${put_price:.4f} ± ${put_se:.4f}")

    # Value at risk
    var_95 = engine.calculate_var(confidence_level=0.95)
    var_99 = engine.calculate_var(confidence_level=0.99)
    print(f"\n  Value at Risk:")
    print(f"   VaR (95%): {var_95*100:.2f}% - loss with probability 5%")
    print(f"   VaR (99%): {var_99*100:.2f}% - loss with probability 1%")
    
    print("\n" + "=" * 60)
















        
        
 
        
        


# monte_carlo_gpu.py
# Maltsev Andrey
"""
GPU version for Monte Carlo
just review and compare with CPU release
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

import cupy as cp
from numba import cuda
import math
import time
from dataclasses import dataclassЬщтеу

# Import from CPU version
import sys
sys.path.append('.')
from src.monte_carlo_cpu import SimulationParams

@cuda.jit
def monte_carlo_kernel(rng_states, S0,K, T, r, sigma, num_steps, final_prices,
                      call_payoffs, put_payoffs):
    """
    CUDA kernel for MonteCarlo simulation
    Each thread simulates one stock price
    """
    # Get index of the current thread
    inx = cuda.grid(1)

    # Bounds checking
    if idx >= final_prices.shape[0]:
        return

    # Time step
    dt = T / num_step

    # Pre-calculation of coefficient
    drift = (r - 0.5 * sigma *sigma) * dt
    vol = sigma * math.sqrt(dt)

    # Initial price
    price = S0

    # Trajectory simulation (cycle by time steps)
    for step in range(num_steps):
        # Random number generation (we use the built-in CUDA generator)
        # In reality this requires a more complex RNG, this is a simplified version
        z = cuda.random.xoroshiro128p_normal_float32(rng_states, idx)

        # Price update: S(t+1) = S(t) * exp(drift + vol * Z)
        price = price * math.exp(drift + vol * z)

    # Save the final price
    final_prices[idx] = price  

    # Calculate  payoffs for options
    call_payoffs[idx] = max(price - K, 0.0)
    put_payoffs[idx] = max(K - price, 0.0)

class MonteCarloGPU:
    """Monte Carlo Engine with GPU (CuPy + Numba)"""

    def __init__(self, params:SimulationParams):
        self.params = params
        self.final_prices = None
        self.call_payoffs = None
        self.put_payoffs = None

    def simulate_gpu_cupy(self):
        """Simulation using CuPy"""
        print(f" GPU similation (CuPy)...")
        print(f" Simulations: {self.params.num_simulations:,}")

        start = time.time()

        # Options
        S0 = self.params.S0
        T = self.params.T
        r = self.params.r
        sigma = self.params.sigma
        N = self.params.num_simulations
        M = self.params.num_steps

        dt = T / M
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * cp.sqrt(dt)

        # Generate random numbers on GPU
        random_numbers = cp.random.standard_normal((N,M), dtype = cp.float32)

        # Initializing arrays on the GPU
        prices = cp.zeros((N, M + 1), dtype=cp.float32)
        prices[:, 0] = S0

        # Trajections simulation (vector. on GPU)
        for i in range(M):
            prices[:, i + 1] = prices[:, i] * cp.exp(
                drift + vol * random_numbers[:, i]
            )

        # Save results
        self.final_prices = prices[:, -1]

        elapsed = time.time() - start
        print(f" Complited for {elapsed:.4f} sec.")

        return self.final_prices

    def price_european_call_gpu(self, K=None):
        """Evaluate Call option on GPU"""
        if self.final_prices is None:
            self.simulate_gpu_cupy()

        if K is None:
            K = self.params.K

        # All computations on GPU
        payoffs = cp.maximum(self.final_prices - K, 0)
        discount_factor = cp.exp(-self.params.r * self.params.T)
        discounted_payoffs = payoffs * discount_factor

        # Average per GPU
        option_price = float(cp.mean(discounted_payoffs))
        std_error = float(cp.std(discounted_payoffs) / cp.sqrt(self.params.num_simulations))

        return option_price, std_error

    def price_european_put_gpu(self, K=None):
        """Evaluate "Put" option on GPU"""
        if self.final_prices is None:
            self.simulate_gpu_cupy()

        if K is None:
            K = self.params.K

        payoffs = cp.maximum(K - self.final_prices, 0)
        discount_factor = cp.exp(-self.params.r * self.params.T)
        discounted_payoffs = payoffs * discount_factor

        option_price = float(cp.mean(discounted_payoffs))
        std_error = float(cp.std(discounted_payoffs) / cp.sqrt(self.params.num_simulations))
        
        return option_price, std_error

    def calculate_var_gpu(self, confidence_level=0.95):
        """Compute VaR on GPU"""
        if self.final_prices is None:
            self.simulate_gpu_cupy()

        returns = (self.final_prices - self.params.S0) / self.params.S0
        var = -float(cp.percentile(returns, (1 - confidence_level) * 100))

        return var

    def get_statistics_gpu(self):
        """Statistics on GPU"""
        if self.final_prices is None:
            self.simulate_gpu_cupy()

        stats = {
            'mean_price': float(cp.mean(self.final_prices)),
            'median_price': float(cp.median(self.final_prices)),
            'std_price': float(cp.std(self.final_prices)),
            'min_price': float(cp.min(self.final_prices)),
            'max_price': float(cp.max(self.final_prices)),
            'percentile_5': float(cp.percentile(self.final_prices, 5)),
            'percentile_95': float(cp.percentile(self.final_prices, 95))
        }

        return stats 

# Benchmark: CPU vs GPU
if __name__ == "__main__":
    print("=" * 70)
    print("Monte Carlo Simulation - GPU vs CPU Benchmark")
    print("=" * 70)
    
    # Options
    params = SimulationParams(
        S0=100.0,
        K=105.0,
        T=1.0,
        r=0.05,
        sigma=0.25,
        num_simulations=1_000_000,
        num_steps=252
    )

    # ===== CPU version =====
    print("\n  CPU VERSION:")
    print("-" * 70)
    from src.monte_carlo_cpu import MonteCarloEngine
    
    cpu_engine = MonteCarloEngine(params)
    cpu_start = time.time()
    cpu_engine.simulate_stock_prices()
    cpu_time = time.time() - cpu_start
    
    cpu_call, _ = cpu_engine.price_european_call()
    cpu_put, _ = cpu_engine.price_european_put()
    cpu_var = cpu_engine.calculate_var()
    
    print(f"  Time computation: {cpu_time:.4f} sec")

    # ===== GPU Version =====
    print("\n GPU VERSION (CuPy):")
    print("-" * 70)
    
    gpu_engine = MonteCarloGPU(params)
    gpu_start = time.time()
    gpu_engine.simulate_gpu_cupy()
    gpu_time = time.time() - gpu_start
    
    gpu_call, _ = gpu_engine.price_european_call_gpu()
    gpu_put, _ = gpu_engine.price_european_put_gpu()
    gpu_var = gpu_engine.calculate_var_gpu()
    
    print(f"  Time computation: {gpu_time:.4f} sec")
    
    #===== Results =====
    print("\n" + "=" * 70)
    print(" COMPARASION RESULTS")
    print("=" * 70)

    speedup = cpu_time / gpu_time
    print(f"\n Speedup: {speedup:.2f}x")
    print(f"  CPU time: {cpu_time:.4f} sec")
    print(f"  GPU time: {gpu_time:.4f} sec")

    print(f"\n Call option:")
    print(f"  CPU: ${cpu_call:.4f}")
    print(f"  GPU: ${gpu_call:.4f}")
    print(f"  Difference: ${abs(cpu_call - gpu_call):.4f}")

    print(f"\n Put option:")
    print(f"  CPU: ${cpu_put:.4f}")
    print(f"  GPU: ${gpu_put:.4f}")
    print(f"  Difference: ${abs(cpu_put - gpu_put):.4f}")

    print(f"\n VaR (95%):")
    print(f"  CPU: {cpu_var*100:.2f}%")
    print(f"  GPU: {gpu_var*100:.2f}%")

    print("\n" + "=" * 70)





























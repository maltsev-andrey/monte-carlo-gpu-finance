# Maltsev Andrey
"""
Monte Carlo Option Pricing - Optimized GPU Implementation (Fast Version)
========================================================================
Author: Andrey Maltsev
GPU: Tesla P100 (Compute Capability 6.0, Pascal Architecture)

Fixed version with faster random number generation
"""

import numpy as np
import cupy as cp
from numba import cuda
import math
import time
from dataclasses import dataclass
from typing import Tuple, Optional

# Configure memory pool for faster allocations
mempool = cp.cuda.MemoryPool()
cp.cuda.set_allocator(mempool.malloc)

@dataclass
class SimulationParams:
    """Parameters optimized for Tesla P100"""
    S0: float = 100.0    # Initial stock price
    K: float = 105.0     # Strike price
    T: float = 1.0       # Time to maturity (years)
    r: float = 0.05      # Risk-free rate
    sigma: float = 0.25  # Volatility
    num_simulations: int = 1_000_000
    num_steps: int = 252  # Trading days
    batch_size: int = 100_000     # Optimal batch for 16GB HBM2 memory
    threads_per_block: int = 256  # CUDA threads

class MonteCarloGPU:
    """Optimized Monte Carlo engine for Tesla P100"""

    def __init__(self, params: SimulationParams):
        self.params = params
        self.configure_gpu()

        # Pre-calculate constants
        self.dt = params.T / params.num_steps
        self.drift = (params.r - 0.5 * params.sigma**2) * self.dt
        self.diffusion = params.sigma * math.sqrt(self.dt)

        # Setup random number generator
        self.rng = cp.random.RandomState(seed=42)

    def configure_gpu(self):
        """Configure GPU for maximum performance"""
        device = cp.cuda.Device()
        
        # Get device properties
        total_memory = device.mem_info[1]
        free_memory = device.mem_info[0]
        
        print("="*60)
        print("TESLA P100 GPU CONFIGURATION")
        print("="*60)
        print(f"Device: {device}")
        print(f"Total Memory: {total_memory / (1024**3):.1f} GB")
        print(f"Free Memory: {free_memory / (1024**3):.1f} GB")
        print(f"Compute Capability: 6.0 (Pascal)")
        print("="*60)

        # Calculate optimal configuration
        self.threads_per_block = self.params.threads_per_block
        self.blocks_per_grid = (self.params.num_simulations + self.threads_per_block - 1) // self.threads_per_block
        
        print(f"Thread Configuration: {self.blocks_per_grid} blocks x {self.threads_per_block} threads")
        print("="*60)

    def simulate_paths_gpu(self) -> cp.ndarray:
        """
        Simulate stock paths using GPU with optimized batching
        """
        N = self.params.num_simulations
        M = self.params.num_steps
        S0 = self.params.S0
        batch_size = self.params.batch_size

        print(f"\nSimulating {N:,} paths with {M} steps each...")
        print(f"Batch size: {batch_size:,}")
        
        # Pre-allocate output array
        all_final_prices = cp.zeros(N, dtype=cp.float32)
        
        # Process in batches to manage memory
        num_batches = (N + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, N)
            current_batch_size = batch_end - batch_start
            
            if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                print(f"  Processing batch {batch_idx + 1}/{num_batches} "
                      f"({batch_start:,} - {batch_end:,}) - {elapsed:.1f}s")
            
            # Generate random numbers for entire batch at once
            randn = self.rng.standard_normal((current_batch_size, M), dtype=cp.float32)
            
            # Calculate cumulative sum of log returns
            log_returns = self.drift + self.diffusion * randn
            cum_log_returns = cp.cumsum(log_returns, axis=1)
            
            # Calculate final prices for all paths in batch
            final_prices = S0 * cp.exp(cum_log_returns[:, -1])
            
            # Store results
            all_final_prices[batch_start:batch_end] = final_prices
            
            # Clear intermediate arrays to free memory
            del randn, log_returns, cum_log_returns, final_prices
            
        total_time = time.time() - start_time
        print(f"\nSimulation completed in {total_time:.2f} seconds")
        print(f"Throughput: {N/total_time:,.0f} paths/second")
        
        return all_final_prices

    def simulate_paths_gpu_fast(self) -> cp.ndarray:
        """
        Even faster version - generate all paths at once if memory allows
        """
        N = self.params.num_simulations
        M = self.params.num_steps
        S0 = self.params.S0
        
        print(f"\nFast simulation: {N:,} paths with {M} steps each...")
        
        # Check if we can fit everything in memory
        bytes_needed = N * M * 4  # float32
        free_memory = cp.cuda.Device().mem_info[0]
        
        if bytes_needed < free_memory * 0.8:  # Use 80% of free memory
            print("  Using single-batch processing (fastest)")
            start_time = time.time()
            
            # Generate all random numbers at once
            randn = self.rng.standard_normal((N, M), dtype=cp.float32)
            
            # Calculate log returns
            log_returns = self.drift + self.diffusion * randn
            
            # Calculate cumulative sum along time axis
            cum_log_returns = cp.cumsum(log_returns, axis=1)
            
            # Get final prices
            final_prices = S0 * cp.exp(cum_log_returns[:, -1])
            
            total_time = time.time() - start_time
            print(f"  Completed in {total_time:.2f} seconds")
            print(f"  Throughput: {N/total_time:,.0f} paths/second")
            
            return final_prices
        else:
            print("  Using batched processing (memory limited)")
            return self.simulate_paths_gpu()

    def price_options(self, final_prices: cp.ndarray) -> dict:
        """
        Price European options and calculate Greeks
        """
        K = self.params.K
        r = self.params.r
        T = self.params.T
        
        # Discount factor
        discount = cp.exp(-r * T)
        
        # Calculate payoffs
        call_payoffs = cp.maximum(final_prices - K, 0)
        put_payoffs = cp.maximum(K - final_prices, 0)
        
        # Calculate prices
        call_price = float(discount * cp.mean(call_payoffs))
        put_price = float(discount * cp.mean(put_payoffs))
        
        # Calculate standard errors
        n = len(final_prices)
        call_std = float(discount * cp.std(call_payoffs) / cp.sqrt(n))
        put_std = float(discount * cp.std(put_payoffs) / cp.sqrt(n))
        
        # Simple Greeks calculation
        itm_calls = final_prices > K
        delta = float(cp.mean(itm_calls))
        
        return {
            'european_call': {
                'price': call_price,
                'std_error': call_std,
                'confidence_interval': (call_price - 1.96*call_std, call_price + 1.96*call_std)
            },
            'european_put': {
                'price': put_price,
                'std_error': put_std,
                'confidence_interval': (put_price - 1.96*put_std, put_price + 1.96*put_std)
            },
            'greeks': {
                'delta': delta,
            }
        }

    def benchmark(self):
        """Run performance benchmark"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Warmup
        print("Warming up GPU...")
        _ = self.rng.standard_normal(1000)
        cp.cuda.Stream.null.synchronize()
        
        configurations = [
            {'name': 'Small batch (10K)', 'batch_size': 10_000},
            {'name': 'Medium batch (100K)', 'batch_size': 100_000},
            {'name': 'Large batch (500K)', 'batch_size': 500_000},
            {'name': 'Single batch (if possible)', 'batch_size': self.params.num_simulations},
        ]
        
        results = []
        
        for config in configurations:
            self.params.batch_size = config['batch_size']
            print(f"\nTesting: {config['name']}")
            
            start = time.time()
            if config['batch_size'] >= self.params.num_simulations:
                final_prices = self.simulate_paths_gpu_fast()
            else:
                final_prices = self.simulate_paths_gpu()
            gpu_time = time.time() - start
            
            throughput = self.params.num_simulations / gpu_time
            
            results.append({
                'config': config['name'],
                'time': gpu_time,
                'throughput': throughput,
                'speedup': 16.12 / gpu_time  # Assuming 16.12s CPU baseline
            })
            
            print(f"  Time: {gpu_time:.3f}s")
            print(f"  Throughput: {throughput:,.0f} paths/sec")
            print(f"  Speedup vs CPU: {results[-1]['speedup']:.1f}x")
        
        # Find best configuration
        best = max(results, key=lambda x: x['throughput'])
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Best configuration: {best['config']}")
        print(f"Best time: {best['time']:.3f} seconds")
        print(f"Best throughput: {best['throughput']:,.0f} paths/second")
        print(f"Maximum speedup: {best['speedup']:.1f}x vs CPU")
        
        return results

# Custom CUDA kernel for even better performance
@cuda.jit
def monte_carlo_kernel(final_prices, randn, S0, drift, diffusion, N, M):
    """
    CUDA kernel optimized for Tesla P100
    Each thread simulates one complete path
    """
    idx = cuda.grid(1)
    
    if idx >= N:
        return
    
    price = S0
    
    # Each thread processes its own path
    for step in range(M):
        # Coalesced memory access pattern
        z = randn[idx * M + step]
        price *= math.exp(drift + diffusion * z)
    
    final_prices[idx] = price

def run_cuda_kernel_version(params):
    """
    Run simulation using custom CUDA kernel
    """
    print("\n" + "="*60)
    print("CUDA KERNEL VERSION")
    print("="*60)
    
    N = params.num_simulations
    M = params.num_steps
    S0 = params.S0
    
    # Pre-calculate constants
    dt = params.T / params.num_steps
    drift = (params.r - 0.5 * params.sigma**2) * dt
    diffusion = params.sigma * math.sqrt(dt)
    
    # Allocate device memory
    d_final_prices = cuda.device_array(N, dtype=np.float32)
    d_randn = cuda.device_array(N * M, dtype=np.float32)
    
    # Generate random numbers
    print("Generating random numbers...")
    randn_host = np.random.standard_normal(N * M).astype(np.float32)
    d_randn = cuda.to_device(randn_host)
    
    # Configure kernel launch
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    print(f"Launching kernel: {blocks_per_grid} blocks x {threads_per_block} threads")
    
    # Launch kernel
    start = time.time()
    monte_carlo_kernel[blocks_per_grid, threads_per_block](
        d_final_prices, d_randn, S0, drift, diffusion, N, M
    )
    cuda.synchronize()
    kernel_time = time.time() - start
    
    # Copy results back
    final_prices = d_final_prices.copy_to_host()
    
    print(f"Kernel execution time: {kernel_time:.3f} seconds")
    print(f"Throughput: {N/kernel_time:,.0f} paths/second")
    print(f"Speedup vs CPU: {16.12/kernel_time:.1f}x")
    
    return cp.asarray(final_prices)

# Main execution
if __name__ == "__main__":
    # Create simulation parameters
    params = SimulationParams(
        S0=100.0,
        K=105.0,
        T=1.0,
        r=0.05,
        sigma=0.25,
        num_simulations=1_000_000,
        num_steps=252,
        batch_size=100_000
    )
    
    # Initialize engine
    print("Initializing Monte Carlo GPU Engine...")
    engine = MonteCarloGPU(params)
    
    # Run fast simulation
    print("\n" + "="*60)
    print("RUNNING OPTIMIZED SIMULATION")
    print("="*60)
    final_prices = engine.simulate_paths_gpu_fast()
    
    # Price options
    results = engine.price_options(final_prices)
    
    print("\n" + "="*60)
    print("OPTION PRICING RESULTS")
    print("="*60)
    
    for option_type, values in results.items():
        if option_type != 'greeks':
            print(f"\n{option_type.upper()}:")
            print(f"  Price: ${values['price']:.4f}")
            print(f"  Std Error: ${values['std_error']:.4f}")
            ci = values['confidence_interval']
            print(f"  95% CI: (${ci[0]:.4f}, ${ci[1]:.4f})")
    
    print(f"\nDelta: {results['greeks']['delta']:.4f}")
    
    # Run benchmark
    benchmark_results = engine.benchmark()
    
    # Try CUDA kernel version if numba is available
    try:
        kernel_prices = run_cuda_kernel_version(params)
        kernel_results = engine.price_options(kernel_prices)
        print(f"\nKernel Call Price: ${kernel_results['european_call']['price']:.4f}")
    except Exception as e:
        print(f"\nCUDA kernel version failed: {e}")
    
    print("\n" + "="*60)
    print("✓ SIMULATION COMPLETE!")
    print("="*60)
# Monte Carlo Option Pricing with GPU Acceleration

Project Overview

High-performance implementation of Monte Carlo method for option pricing using GPU acceleration through CUDA. The project shows computational speedup when transitioning from CPU to GPU.

## Performance

| Version    | Time (sec) | Speedup | Accuracy |
|------------|------------|---------|----------|
| CPU        | 16.12      | 1.0x    | Baseline |
| GPU (CuPy) | 1.22       | 13.24x  | ±$0.04  |

### Test Configuration
- **GPU**:	Tesla P100-PCIE-16GB
- **CUDA**: 	12.4
- **Driver**: 	550.90.07
- **CPU**: 	Intel Xeon E5-2678 v3@2.50GHz

## Features

- **Real Market Data**:     Loading via yfinance API
- **European Options**:     Call and Put options
- **Risk Metrics**: 	    Value at Risk (VaR), Greeks
- **GPU Optimization**:     CuPy and Numba CUDA
- **Comparative Analysis**: Detailed CPU vs GPU comparison
- **Visualization**: 	    Distribution and result charts

## Installation
````
### Requirements
- Python 3.9+
- CUDA Toolkit 11.0+ 
- CUDA-capable GPU

### Installing Dependencies
----------------------------
# Clone repository
git clone https://github.com/maltsev-andrey/monte-carlo-gpu-finance.git
cd monte-carlo-gpu-finance

# Install dependencies
pip install -r requirements.txt

# For GPU
pip install cupy-cuda12x
----------------------------
````

## Usage
```
from src.monte_carlo_cpu import MonteCarloEngine, SimulationParams
from src.monte_carlo_gpu import MonteCarloGPU

# Simulation parameters
params = SimulationParams(
    S0=100.0,        # Initial price
    K=105.0,         # Strike price
    T=1.0,           # Time to maturity (years)
    r=0.05,          # Risk-free rate
    sigma=0.25,      # Volatility
    num_simulations=1_000_000,  # Number of simulations
    num_steps=252    # Time steps (trading days)
)
```

# CPU version
cpu_engine = MonteCarloEngine(params)
cpu_engine.simulate_stock_prices()
call_price, _ = cpu_engine.price_european_call()

# GPU version
gpu_engine = MonteCarloGPU(params)
gpu_engine.simulate_gpu_cupy()
call_price, _ = gpu_engine.price_european_call_gpu()


### Running Benchmarks
----------------------------
# CPU vs GPU comparison
python src/monte_carlo_gpu.py

# Scalability testing
python benchmark_scaling.py

# GPU profiling
python profile_gpu.py

## Project Structure
----------------------------
monte-carlo-gpu-finance/
├── src/
│   ├── monte_carlo_cpu.py      # CPU implementation
│   ├── monte_carlo_gpu.py      # GPU implementation (CuPy)
│   ├── monte_carlo_gpu_naive.py # Basic GPU version
│   ├── monte_carlo_gpu_optimized.py # Optimized version
│   └── data_loader.py          # Market data loader
├── benchmarks/
│   ├── performance_benchmark.py # Performance measurements
│   ├── scaling_test.py          # Scalability tests
│   └── results/                 # Test results
├── notebooks/
│   └── analysis.ipynb           # Jupyter notebook with analysis
├── tests/
│   └── test_accuracy.py        # Accuracy tests
└── requirements.txt
----------------------------

## Measurement Methodology

- **Simulations**:	From 100K to 10M paths
- **Time Steps**:	252 (trading days per year)
- **Repetitions**: 	5 runs for averaging
- **Metrics**: 		Execution time, memory usage, accuracy

### Time Measurement
```
# CPU timing
start_time = time.time()
cpu_engine.simulate_stock_prices()
cpu_time = time.time() - start_time

# GPU timing (with synchronization)
start_time = time.time()
gpu_engine.simulate_gpu_cupy()
cp.cuda.Stream.null.synchronize()  # Important for accurate measurement
gpu_time = time.time() - start_time
```

## Optimizations
1. **Vectorization**:		Using CuPy for array operations
2. **Memory Coalescing**: 	Optimal GPU memory access
3. **Batch Processing**: 	Processing data in batches
4. **Float32**: 			Using single precision where possible

### Planned Improvements
- [ ] Use shared memory for frequently accessed data
- [ ] CUDA block size optimization
- [ ] Quasi-Monte Carlo implementation (Sobol sequences)
- [ ] American options (Longstaff-Schwartz)
- [ ] Other options (barrier, Asian)

## Results

| Simulations | CPU (sec) | GPU (sec) | Speedup |
|-------------|-----------|-----------|---------|
| 100K        | 1.62      | 0.31      | 5.2x    |
| 500K        | 8.05      | 0.68      | 11.8x   |
| 1M          | 16.12     | 1.22      | 13.2x   |
| 5M          | 80.5      | 3.45      | 23.3x   |
| 10M         | 161.2     | 6.21      | 26.0x   |

### Accuracy
- **Call Option**:	Difference < 0.5% from analytical Black-Scholes solution
- **Put Option**:	Difference < 0.5%
- **VaR (95%)**:	Agreement within 0.1%

## License

MIT License - see [LICENSE](LICENSE) file for details

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [Monte Carlo Methods in Finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance)
- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

## Author
**Andrey Maltsev**
- GitHub: [@maltsev-andrey](https://github.com/maltsev-andrey)
- Email: andrey.maltsev@yahoo.com

---

⭐ If you found this project helpful, please give it a star on GitHub!


## Latest Benchmark Results

See [detailed benchmark results](benchmarks/BENCHMARK_RESULTS.md) for performance analysis.



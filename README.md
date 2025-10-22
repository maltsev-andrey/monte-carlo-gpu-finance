# Monte Carlo Option Pricing - GPU Accelerated 🚀

[![GPU](https://img.shields.io/badge/GPU-Tesla%20P100-76B900.svg)](https://www.nvidia.com/en-us/data-center/tesla-p100/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Performance](https://img.shields.io/badge/Speedup-25%2C476x-red.svg)](https://github.com/yourusername/monte-carlo-gpu)

## Project Overview

Ultra-optimized Monte Carlo simulation for option pricing using NVIDIA Tesla P100 GPU. Achieved **25,476x speedup** over CPU implementation through advanced GPU optimization techniques.

### Key Achievements
- **Performance**: 1.58 billion paths/second throughput
- **Speed**: Reduced computation from 16.12s to 0.001s  
- **Efficiency**: Near-peak HBM2 memory bandwidth utilization (732 GB/s)
- **Scale**: Processes 1M paths × 252 time steps in milliseconds

## Architecture

```
Tesla P100 Specifications:
- Architecture: Pascal (Compute Capability 6.0)
- Memory: 16GB HBM2 @ 732 GB/s
- CUDA Cores: 3584 (56 SMs × 64 cores)
- FP32 Performance: 9.3 TFLOPS
- Power: 250W TDP
```

## Performance Benchmarks

| Configuration    | Time (s) | Throughput (paths/sec) | Speedup     |
|------------------|----------|------------------------|-------------|
| CPU Baseline     | 16.120   | 62K                    | 1x          |
| GPU (10K batch)  | 0.026    | 38M                    | 614x        |
| GPU (100K batch) | 0.003    | 356M                   | 5,740x      |
| GPU (500K batch) | 0.001    | 1.39B                  | 22,359x     |
| **GPU (Optimal)**| **0.001**| **1.58B**              | **25,476x** |

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support (tested on Tesla P100)
- CUDA Toolkit 12.0+ 
- Python 3.9+

### Setup
```bash
# Clone repository
git clone https://github.com/maltsev-andrey/monte-carlo-gpu.git
cd monte-carlo-gpu

# Install dependencies
pip install numpy cupy-cuda12x numba

# Verify GPU
python -c "import cupy; print(cupy.cuda.Device())"
```

## Usage

### Quick Start
```python
from monte_carlo_optimized_v2 import MonteCarloGPU, SimulationParams

# Configure simulation
params = SimulationParams(
    S0=100.0,      # Initial stock price
    K=105.0,       # Strike price
    T=1.0,         # Time to maturity (years)
    r=0.05,        # Risk-free rate
    sigma=0.25,    # Volatility
    num_simulations=1_000_000,
    num_steps=252  # Trading days
)

# Run simulation
engine = MonteCarloGPU(params)
final_prices = engine.simulate_paths_gpu_fast()
results = engine.price_options(final_prices)

# Display results
print(f"Call Price: ${results['european_call']['price']:.4f}")
print(f"Put Price: ${results['european_put']['price']:.4f}")
```

### Run Complete Benchmark
```bash
python src/monte_carlo_optimized_v2.py
```

## Project Structure

```
monte-carlo-gpu/
├── src/
│   ├── monte_carlo_optimited.py       # Original version (v1.0)
│   └── monte_carlo_optimized_v2.py    # Optimized version (v2.0)
├── docs/
│   └── optimization_changelog.txt     # Detailed optimization log
├── benchmarks/
│   └── results.json                   # Benchmark results
├── requirements.txt
└── README.md
```

## Optimization Techniques

### 1. Vectorization
- Replaced sequential loops with GPU-accelerated array operations
- Utilized CuPy's optimized CUDA libraries (cuBLAS, cuRAND)

### 2. Memory Management  
- Single-batch processing when memory permits
- Intelligent batching for larger simulations
- Pre-allocation and immediate cleanup of arrays

### 3. Algorithm Optimization
- Removed inefficient Sobol sequence generation
- Used cumulative sum for path generation
- Leveraged GPU's parallel architecture

## Options Pricing Validation

```
Test Parameters:
- Stock Price: $100, Strike: $105, Maturity: 1 year
- Risk-free Rate: 5%, Volatility: 25%

Results (1M simulations):
- European Call: $10.0177 ± 0.0170
- European Put:  $9.8727 ± 0.0126  
- Delta: 0.4532

Validated against Black-Scholes analytical solution
```

## Use Cases

- **High-Frequency Trading**: Real-time option pricing
- **Risk Management**: Large-scale portfolio simulations
- **Research**: Financial modeling and analysis
- **Education**: GPU computing and quantitative finance

## Version History

### v2.0 (October 2024)
- Fixed critical recursion bug
- Removed slow Sobol implementation
- Achieved 25,476x speedup
- Added comprehensive benchmarking

### v1.0 (October 2024)  
- Initial implementation
- Tesla P100 optimizations
- FP16 support (experimental)

## Scalability

| Paths | Memory Required | Batch Strategy | Expected Time |
|-------|-----------------|----------------|---------------|
| 1M    | 1 GB            | Single         | 0.001s        |
| 10M   | 10 GB           | Single         | 0.01s         |
| 100M  | 100 GB          | 10 batches     | 0.1s          |
| 1B    | 1 TB            | 100 batches    | 1s            |

## Contributing

Contributions welcome! Areas for improvement:
- Multi-GPU support
- American option pricing
- Variance reduction techniques
- Alternative random number generators

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Author**: Andrey Maltsev
- **Hardware**: NVIDIA Tesla P100 GPU

## Contact

- GitHub: [@maltsev-andrey](https://github.com/maltsev-andrey)
- Email: andrey.maltsev@yahoo.com

---

** Performance Achievement: 25,476x Faster Than CPU**  
*From 16.12 seconds → 0.001 seconds*


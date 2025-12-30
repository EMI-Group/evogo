<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/_static/evox_logo_light.png">
    <img alt="EvoX Logo" height="50" src="docs/images/evox_logo_light.png">
  </picture>
</h1>

<h2 align="center">
🌟 Evolutionary Generative Optimization: Towards Fully Data-Driven Evolutionary Optimization via Generative Learning 🌟
</h2>

<div align="center">
  <a href="https://www.arxiv.org/abs/2508.00380">
    < img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="EvoGO Paper on arXiv">
  </a >
</div>

## 📚 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [JAX Version](#jax-version)
  - [Installation (JAX)](#installation)
  - [Example (JAX)](#example)
- [Torch Version](#torch-version)
  - [Installation (Torch)](#installation-1)
  - [Example (Torch)](#example-1)
- [Community & Support](#-community--support)
- [Citing EvoGO](#-citing-evogo)


## 🔍 Overview

**EvoGO (Evolutionary Generative Optimization)** is a **fully data-driven framework for black-box optimization** designed to overcome a key bottleneck of traditional evolutionary optimization—its reliance on **operator-level manual design** (e.g., crossover and mutation rules), which limits adaptability and makes performance heavily dependent on tuning and domain knowledge. By shifting the optimization paradigm toward **learning-driven search behavior from historical evaluations**, EvoGO reduces heuristic dependency and improves robustness across complex optimization scenarios. EvoGO supports both **JAX and PyTorch** backends and is designed to be compatible with **EvoX**.


## ✨ Key Features

### 🚫 Fully Data-Driven Optimization (Replacing Handcrafted Operators)
- Replaces operator-centric evolutionary design with learned generation to drive the search process.
- Reduces reliance on manual heuristics and extensive parameter tuning.


### 🎯 Optimization-Goal Alignment
- Introduces learning objectives explicitly tailored for optimization, improving the consistency between model training and search goals.
- Helps mitigate common failure modes in learning-based optimization such as premature convergence and misdirected generation.

### ⚡ Scalable and Parallel-Friendly Search
- Naturally supports large-scale parallel candidate generation and evaluation.
- Well-suited for modern high-throughput settings such as GPU-accelerated simulation-based optimization.

### 🧪 Strong Performance across Diverse Black-Box Benchmarks
- Validated on numerical benchmarks, classical control tasks, and high-dimensional robotic control environments.
- Demonstrates fast convergence and competitive wall-clock efficiency compared with representative baselines.

## JAX Version

### Installation

To set up the JAX environment, simply run the provided setup script:

```bash
bash setup_env_jax.sh
```

### Example

Here is an example of how to use EvoGo with the JAX backend:

```python
import sys
import os
import jax
import jax.numpy as jnp

# 1. Add project root to sys.path to enable importing evogo_jax
# Assuming this script is in example/jax/ directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from evogo_jax.evogo import EvoGO
from test_functions.simple_functions_jax import Rosenbrock

if __name__ == "__main__":
    # 2. Configuration
    dim = 5
    num_parallel = 2
    seed = 42
    key = jax.random.PRNGKey(seed)

    # 3. Initialize the objective function from test_functions
    # Rosenbrock function expects input in [0, 1] and scales it internally
    objective_fn = Rosenbrock(dim=dim, key=key, parallels=num_parallel)

    # 4. Initialize the solver
    solver = EvoGO(
        max_iter=5,          # Number of iterations
        batch_size=100,      # Initial sample size
        gm_batch_size=100,   # Generative model sample size
        num_parallel=num_parallel, # Number of parallel runs
        debug=True           # Print progress
    )

    # 5. Call the solve method
    print(f"Starting to solve {dim}-dimensional Rosenbrock function...")
    best_x, best_y = solver.solve(objective_fn, dim=dim, seed=seed)

    print("\n" + "="*30)
    print(f"Optimization Complete!")
    print(f"Best x: {best_x}")
    print(f"Best y: {best_y}")
    print("="*30)
```

## Torch Version

### Installation

To set up the PyTorch environment, simply run the provided setup script:

```bash
bash setup_env_torch.sh
```

### Example

Here is an example of how to use EvoGo with the PyTorch backend:

```python
import sys
import os
import torch
import numpy as np

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from evogo_torch.evogo import EvoGO
from test_functions.simple_functions import HarderNumerical

if __name__ == "__main__":
    # Configuration
    dim = 5
    num_parallel = 2
    seed = 42
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device) # HarderNumerical expects torch.device or string? 
    # checking simple_functions.py: re_init takes torch.device. __init__ takes torch.device.
    device_obj = torch.device(device)
    
    print(f"Running on {device}")

    # Initialize the objective function from test_functions
    # HarderNumerical wraps the function with random offsets and rotations (unless affine=False)
    # We use affine=False to match the simple Rosenbrock behavior if desired, or True for harder version.
    # Let's use defaults (affine=True) as it is "HarderNumerical".
    objective_fn = HarderNumerical(
        dim=dim, 
        device=device_obj, 
        eval_fn=HarderNumerical.Rosenbrock, 
        instances=num_parallel
    )

    # Initialize the solver
    solver = EvoGO(
        max_iter=5,
        batch_size=100,
        gm_batch_size=100,
        num_parallel=num_parallel,
        use_gp=True, # Use Gaussian Process
        debug=True,
        gpu_id=0 if "cuda" in device else -1
    )

    # Call the solve method
    print(f"Starting to solve {dim}-dimensional Rosenbrock function (HarderNumerical)...")
    
    best_x, best_y = solver.solve(objective_fn, dim=dim, seed=seed, device=device)

    print("\n" + "="*30)
    print(f"Optimization Complete!")
    print(f"Best x (normalized input to solver): {best_x}")
    print(f"Best y: {best_y}")
    print("="*30)
```

### 🤝 Community & Support

- GitHub Issues for bug reports and feature requests

- Contributions are welcome via pull requests

- Please include minimal reproducible examples for faster debugging

### 📖 Citing EvoGO

```
@article{sun2025evolutionary,
  title     = {Evolutionary Generative Optimization: Towards Fully Data-Driven Evolutionary Optimization via Generative Learning},
  author    = {Sun, Kebin and Jiang, Tao and Cheng, Ran and Jin, Yaochu and Tan, Kay Chen},
  journal   = {arXiv preprint arXiv:2508.00380},
  year      = {2025}
}

```

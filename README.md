<h1 align="center">

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/evox_logo_light.png">
    <img alt="EvoX Logo" height="50" src="docs/images/evox_logo_light.png">
  </picture>


</h1>

<h2 align="center">
🌟 Evolutionary Generative Optimization: Towards Fully Data-Driven Evolutionary Optimization via Generative Learning 🌟
</h2>

<div align="center">
  <a href="https://www.arxiv.org/abs/2508.00380">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="EvoGO Paper on arXiv">
  </a>
</div>

<div align="center">
  <img src="docs/papers/overview.png" alt="EvoGO Overview" width="90%">
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

Evolutionary Generative Optimization(EvoGO) is a fully data-driven framework for black-box optimization.
It addresses a core limitation of traditional evolutionary optimization that relies on manually designed operators.
EvoGO replaces heuristic rules by learning search behavior from historical evaluations.
The framework supports both JAX and PyTorch backends.
EvoGO is designed to be compatible with EvoX.


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
import jax
import jax.numpy as jnp

from evogo_jax.evogo import EvoGO
from test_functions.simple_functions_jax import Rosenbrock

if __name__ == "__main__":
    # 2. Configuration
    dim = 5
    num_parallel = 2
    seed = 42
    key = jax.random.PRNGKey(seed)

    # 3. Initialize the problem
    problem = Rosenbrock(dim=dim, key=key, parallels=num_parallel)

    # 4. Initialize the algorithm
    solver = EvoGO(
        max_iter=5,          
        batch_size=100,     
        gm_batch_size=100,  
        num_parallel=num_parallel, 
        debug=True           
    )

    # 5. Call the solve method
    print(f"Starting to solve {dim}-dimensional Rosenbrock function...")
    best_x, best_y = solver.solve(problem, dim=dim, seed=seed)

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
import torch
import numpy as np

from evogo_torch.evogo import EvoGO
from test_functions.simple_functions import HarderNumerical

if __name__ == "__main__":
    # Configuration
    dim = 5
    num_parallel = 2
    seed = 42
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    print(f"Running on {device}")

    # Initialize the problem
    problem = HarderNumerical(
        dim=dim, 
        device=device_obj, 
        eval_fn=HarderNumerical.Rosenbrock, 
        instances=num_parallel
    )

    # Initialize the algorithm
    solver = EvoGO(
        max_iter=5,
        batch_size=100,
        gm_batch_size=100,
        num_parallel=num_parallel,
        use_gp=True, 
        debug=True,
        gpu_id=0 if "cuda" in device else -1
    )

    # Call the solve method
    print(f"Starting to solve {dim}-dimensional Rosenbrock function (HarderNumerical)...")
    
    best_x, best_y = solver.solve(problem, dim=dim, seed=seed, device=device)

    print("\n" + "="*30)
    print(f"Optimization Complete!")
    print(f"Best x (normalized input to solver): {best_x}")
    print(f"Best y: {best_y}")
    print("="*30)
```

### 🤝 Community & Support

We welcome contributions and look forward to your feedback!

- Engage in discussions and share your experiences on [GitHub Issues](https://github.com/EMI-Group/evogo/issues).
- Join our QQ group (ID: 297969717).

### 📖 Citing EvoGO

```
@article{sun2025evolutionary,
  title     = {Evolutionary Generative Optimization: Towards Fully Data-Driven Evolutionary Optimization via Generative Learning},
  author    = {Sun, Kebin and Jiang, Tao and Cheng, Ran and Jin, Yaochu and Tan, Kay Chen},
  journal   = {arXiv preprint arXiv:2508.00380},
  year      = {2025}
}

```

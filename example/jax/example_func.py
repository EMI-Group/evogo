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

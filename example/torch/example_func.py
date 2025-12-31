import sys
import os
import torch
import numpy as np

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from evogo_torch.evogo import EvoGO
from test_functions.simple_functions import Rosenbrock

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
    # Rosenbrock function expects input in [0, 1] and scales it internally
    objective_fn = Rosenbrock(
        dim=dim, 
        device=device_obj, 
        parallels=num_parallel
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
    print(f"Starting to solve {dim}-dimensional Rosenbrock function...")
    
    best_x, best_y = solver.solve(objective_fn, dim=dim, seed=seed, device=device)

    print("\n" + "="*30)
    print(f"Optimization Complete!")
    print(f"Best x (normalized input to solver): {best_x}")
    print(f"Best y: {best_y}")
    print("="*30)

import os
import sys
import torch
import numpy as np
from typing import Callable, Tuple, Optional, List

from evogo_torch.main import ArgsProtocol
from evogo_torch.step import evogo_step
from evogo_torch.utils import latin_hyper_cube, print_with_prefix, sort_select
from evogo_torch import trainer
from evogo_torch import models

class EvoGO:
    def __init__(
        self,
        max_iter: int = 10,
        batch_size: int = 1000,
        gm_batch_size: int = 1000,
        num_parallel: int = 1,
        use_gp: bool = True,
        use_mlp: bool = False,
        use_direct: bool = False,
        portion: float = 0.1,
        slide_window: float = 0.3,
        cycle_scale: float = 400,
        cond_dim: int = 0,
        use_inv: bool = True,
        use_fast_dist: bool = True,
        drop_rate: float = 1/128,
        use_gan: bool = False,
        single_gen: bool = False,
        sample_via_model: bool = True,
        compile: bool = False,
        use_lcb: bool = False,
        gpu_id: int = 0,
        output_dir: Optional[str] = None,
        debug: bool = False
    ):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.gm_batch_size = gm_batch_size
        self.num_parallel = num_parallel
        self.use_gp = use_gp
        self.use_mlp = use_mlp
        self.use_direct = use_direct
        self.portion = portion
        self.slide_window = slide_window
        self.cycle_scale = cycle_scale
        self.cond_dim = cond_dim
        self.use_inv = use_inv
        self.use_fast_dist = use_fast_dist
        self.drop_rate = drop_rate
        self.use_gan = use_gan
        self.single_gen = single_gen
        self.sample_via_model = sample_via_model
        self.compile = compile
        self.use_lcb = use_lcb
        self.gpu_id = gpu_id
        self.debug = debug
        
        if output_dir is None:
            self.output_dir = f"results_solver_torch_p{use_mlp:d}{use_direct:d}{use_gp:d}"
        else:
            self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_args(self, func_id: int = 0) -> ArgsProtocol:
        """Create ArgsProtocol object from class attributes."""
        return ArgsProtocol(
            gpu_id=self.gpu_id,
            out=self.output_dir,
            save_iter=None, # Not exposing save_iter for now in simple interface
            save_count=-1,
            max_iter=self.max_iter,
            func_id=func_id,
            num_parallel=self.num_parallel,
            repeats=1,
            force_repeat=-1,
            batch_size=self.batch_size,
            gm_batch_size=self.gm_batch_size,
            use_inv=self.use_inv,
            sample_via_model=self.sample_via_model,
            use_fast_dist=self.use_fast_dist,
            compile=self.compile,
            use_gp=self.use_gp,
            use_mlp=self.use_mlp,
            use_direct=self.use_direct,
            use_gan=self.use_gan,
            use_single_gen=self.single_gen,
            use_lcb=self.use_lcb,
            portion=self.portion,
            slide_window=self.slide_window,
            cycle_scale=self.cycle_scale,
            drop_rate=self.drop_rate
        )

    def solve(
        self,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        seed: int = 0,
        device: str = "cuda:0"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the optimization problem.
        
        Args:
            objective_fn: The function to minimize. Input shape (num_parallel, batch_size, dim), 
                          output shape (num_parallel, batch_size).
            dim: Dimension of the input x.
            seed: Random seed.
            device: Device to run on.
            
        Returns:
            Best x found and its corresponding y value.
        """
        torch.manual_seed(seed)
        torch.set_default_device(device)
        
        # Set global flags in models/trainer
        models.USE_INV = self.use_inv
        models.USE_FAST_DIST = self.use_fast_dist
        trainer.COMPILE = self.compile

        # Create args for evogo_step
        args = self._create_args()
        
        # Initial sampling
        datasets_x = latin_hyper_cube(self.num_parallel, self.batch_size, dim, device=torch.device(device))
        datasets_y = objective_fn(datasets_x)
        
        histories = None
        mins_y = torch.min(datasets_y, dim=1).values
        
        for step in range(1, self.max_iter + 1):
            if self.debug:
                print(f"[INFO]  Start step {step}/{self.max_iter}, current best value = {print_with_prefix(mins_y)}")
            
            datasets_x, datasets_y, histories = evogo_step(
                args,
                objective_fn, 
                datasets_x,
                datasets_y,
                histories,
                self.debug,
                None # SAVE
            )
            
            mins_y = torch.min(datasets_y, dim=1).values
            if self.debug:
                print(f"[INFO]  End step {step}/{self.max_iter}, current best value = {print_with_prefix(mins_y)}")
                
        # Return the best result across all parallels
        # datasets_y shape: [num_parallel, total_samples]
        # datasets_x shape: [num_parallel, total_samples, dim]
        
        best_vals, best_indices = torch.min(datasets_y, dim=1) # [num_parallel]
        best_parallel_idx = torch.argmin(best_vals)
        best_sample_idx = best_indices[best_parallel_idx]
        
        best_x = datasets_x[best_parallel_idx, best_sample_idx]
        best_y = datasets_y[best_parallel_idx, best_sample_idx]
        
        return best_x, best_y

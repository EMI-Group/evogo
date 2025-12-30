from typing import Tuple, Callable, List, Optional, Any
import math
import os
import sys
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

# Ensure the parent directory is in sys.path for imports
if os.path.abspath('.') not in sys.path:
    sys.path.append(os.path.abspath('.'))

import evogo_jax.models as models
import evogo_jax.data_loader as data_loader
from evogo_jax.data_loader import train_model
from evogo_jax.utils import latin_hyper_cube, print_with_prefix, sort_select
from evogo_jax.models import (MLPBase, VAELoss, MaternGaussianProcess, GenerativeModel, PairedGenerativeLoss, MLPPredictor,
                        MLPPredictionLoss, GpMllNetwork, get_gp_eval_fn, get_gp_eval_pair_diff_fn, data_split, denormalize, normalize_with,
                        standardize_with)

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
        mip_scale: float = 0.1,
        cond_dim: int = 0,
        use_inv: bool = True,
        use_fast_dist: bool = True,
        drop_rate: float = 1/128,
        use_wide: bool = False,
        use_deep: bool = False,
        use_gan: bool = False,
        single_gen: bool = False,
        gm_use_model: bool = True,
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
        self.mip_scale = mip_scale
        self.cond_dim = cond_dim
        self.use_inv = use_inv
        self.use_fast_dist = use_fast_dist
        self.drop_rate = drop_rate
        self.use_wide = use_wide
        self.use_deep = use_deep
        self.use_gan = use_gan
        self.single_gen = single_gen
        self.gm_use_model = gm_use_model
        self.debug = debug
        
        if output_dir is None:
            self.output_dir = f"results_solver_p{use_mlp:d}{use_direct:d}{use_gp:d}"
        else:
            self.output_dir = output_dir
            
        # Set global flags in models/data_loader if necessary
        models.USE_INV = self.use_inv
        models.USE_FAST_DIST = self.use_fast_dist
        # data_loader.GRAD_BREAKPOINT = False # Default

    def gm_step(
        self,
        eval_fn: Callable[[jax.Array], jax.Array],
        key: jax.Array,
        datasets_x: jax.Array,
        datasets_y: jax.Array,
        histories: Optional[Tuple[jax.Array, jax.Array]] = None,
        save_step: Optional[int] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        
        key, key1 = jax.random.split(key)
        # get GP dataset
        (all_datasets_x, all_datasets_y), \
        (wins_x, loses_x, wins_y, loses_y, conds), \
        (de_norm_x, de_std_y) = data_split(key1,
                                           datasets_x,
                                           datasets_y,
                                           histories=histories,
                                           portion=self.portion,
                                           sliding_window=self.slide_window,
                                           condition_dim=self.cond_dim)
        
        dataset_size = all_datasets_x.shape[1]
        gm_batch_size = self.gm_batch_size
        
        # train GP
        key, key1, key2 = jax.random.split(key, num=3)
        if self.use_gp:
            lr = max(dataset_size / 600, 0.1) * 0.01
            gp_net = GpMllNetwork()
            params = gp_net.init(key1, all_datasets_x, all_datasets_y)
            tx = optax.adam(learning_rate=lr)
            opt_state = tx.init(params=params)
            if self.debug: print(f"[INFO]  Training Gaussian Process with learning rate = {lr}...")
            best_gp_params, trained = train_model(key2,
                                                  batch_size=dataset_size,
                                                  epochs=1600,
                                                  net=gp_net,
                                                  tx=tx,
                                                  params=params,
                                                  opt_state=opt_state,
                                                  shuffle=False,
                                                  DEBUG=self.debug,
                                                  valid_portion=0,
                                                  loss_names={},
                                                  all_datasets_x=all_datasets_x,
                                                  all_datasets_y=all_datasets_y)
            if not trained.all():
                print(f"[ERROR] Not trained GP detected, skip current iteration")
                return key, datasets_x, datasets_y, histories
            
            gp_net = MaternGaussianProcess()
            gp_params = {"params": best_gp_params["params"]["gp"]}
            eval_diff_fn = get_gp_eval_pair_diff_fn(gp_net, gp_params, all_datasets_x, all_datasets_y)
            eval_single_fn = get_gp_eval_fn(gp_net, gp_params, all_datasets_x, all_datasets_y)
        elif self.use_mlp:
            lr = 0.05 / all_datasets_x.shape[1]
            mlp_net = MLPPredictionLoss()
            params = mlp_net.init(key1, all_datasets_x, all_datasets_y, training=False)
            tx = optax.adam(learning_rate=lr)
            opt_state = tx.init(params=params)
            if self.debug: print(f"[INFO]  Training MLP Predictor with learning rate = {lr}...")
            best_mlp_params, trained = train_model(key2,
                                                   batch_size=min(64, all_datasets_x.shape[1]),
                                                   epochs=200,
                                                   net=mlp_net,
                                                   tx=tx,
                                                   params=params,
                                                   opt_state=opt_state,
                                                   shuffle=True,
                                                   DEBUG=self.debug,
                                                   valid_portion=0.1,
                                                   loss_names={},
                                                   all_datasets_x=all_datasets_x,
                                                   all_datasets_y=all_datasets_y)
            mlp_net = MLPPredictor()
            best_mlp_params = {"params": best_mlp_params["params"]["net"]}
            
            @jax.jit
            def _mlp_eval_pair_diff(inputs1: jax.Array, inputs2: jax.Array):
                inputs1 = denormalize(inputs1, *de_norm_x)
                inputs2 = denormalize(inputs2, *de_norm_x)
                out1 = mlp_net.apply(best_mlp_params, inputs1, training=False)
                out2 = mlp_net.apply(best_mlp_params, inputs2, training=False)
                out1 = standardize_with(out1, *de_std_y)
                out2 = standardize_with(out2, *de_std_y)
                return out1 - out2, jnp.ones_like(out2)
            
            eval_diff_fn = _mlp_eval_pair_diff
            
            @jax.jit
            def _mlp_eval_single(inputs: jax.Array):
                inputs = denormalize(inputs, *de_norm_x)
                out = mlp_net.apply(best_mlp_params, inputs, training=False)
                return out, jnp.ones_like(out)
            
            eval_single_fn = _mlp_eval_single
        elif self.use_direct:
            @jax.jit
            def _direct_eval_pair_diff(inputs1: jax.Array, inputs2: jax.Array):
                inputs1 = denormalize(inputs1, *de_norm_x)
                inputs2 = denormalize(inputs2, *de_norm_x)
                out1 = eval_fn(inputs1)
                out2 = eval_fn(inputs2)
                out1 = standardize_with(out1, *de_std_y)
                out2 = standardize_with(out2, *de_std_y)
                return out1 - out2, jnp.ones_like(out2)
            
            eval_diff_fn = _direct_eval_pair_diff
            
            @jax.jit
            def _direct_eval_single(inputs: jax.Array):
                inputs = denormalize(inputs, *de_norm_x)
                out = eval_fn(inputs)
                out = standardize_with(out, *de_std_y)
                return out, jnp.ones_like(out)
            
            eval_single_fn = _direct_eval_single
        else:
            eval_diff_fn = None
            eval_single_fn = None

        # train generative model
        pgl_net = PairedGenerativeLoss(eval_diff_fn,
                                       single_gen=self.single_gen,
                                       wide=self.use_wide,
                                       deep=self.use_deep,
                                       gan=self.use_gan,
                                       drop_rate=self.drop_rate,
                                       cycle_scale=self.cycle_scale,
                                       out_scale=1,
                                       mll_scale=1 if self.use_fast_dist else 0.25,
                                       mip_scale=self.mip_scale,
                                       mip_std_scale=1)
        key, key1, key2 = jax.random.split(key, num=3)
        params = pgl_net.init({
            'params': key1,
            'dropout': key1
        },
                              wins_x[:, :gm_batch_size],
                              loses_x[:, :gm_batch_size],
                              wins_y[:, :gm_batch_size],
                              loses_y[:, :gm_batch_size],
                              conds[:, :gm_batch_size],
                              training=False)
        lr = (1500 / min(gm_batch_size, 1000)) * 1e-5
        epochs = 200
        if self.use_mlp or self.use_direct:
            lr = 1e-4
        tx = optax.adam(learning_rate=lr)
        opt_state = tx.init(params=params)
        if self.debug: print(f"[INFO]  Training generative model with batch size = {gm_batch_size}, learning rate = {lr} ...")
        best_params, trained = train_model(key2,
            batch_size=max(min(gm_batch_size, wins_x.shape[1]), 1),
            epochs=int(epochs * max(1000 / max(datasets_x.shape[1], gm_batch_size), 0.25)),
            net=pgl_net,
            tx=tx,
            params=params,
            opt_state=opt_state,
            shuffle=True,
            DEBUG=self.debug,
            valid_portion=0.0,
            loss_names={
                "lose cycle": 0,
                "win cycle": 1,
                "lose out": 2,
                "win out": 3,
                "win - lose": 4,
                "lose - win": 5,
                "MIP": 6,
                "MIP 2": 7
            },
            wins_x=wins_x,
            loses_x=loses_x,
            wins_y=wins_y,
            loses_y=loses_y,
            conds=conds
        )  
        
        # get next step data
        gm: GenerativeModel = pgl_net.apply(best_params, method=pgl_net._get_nets)[1]
        gm_params = {"params": best_params["params"]["lose2win"]}
        key, key1 = jax.random.split(key)
        new_conds = jax.random.normal(key1, shape=(self.num_parallel, datasets_x.shape[1], self.cond_dim))
        histories = (datasets_x, datasets_y)
        new_datasets_x: jax.Array = gm.apply(gm_params,
                                             normalize_with(datasets_x, *de_norm_x),
                                             new_conds,
                                             training=False,
                                             rngs={'dropout': key1})
        
        new_datasets_x = jnp.clip(new_datasets_x, a_min=-0.25, a_max=1.25)
        new_datasets_x = denormalize(new_datasets_x, *de_norm_x)
        new_datasets_x = jnp.clip(new_datasets_x, a_min=0, a_max=1)
        
        new_datasets_y: jax.Array = eval_fn(new_datasets_x)
        datasets_x = jnp.concatenate([datasets_x, new_datasets_x], axis=1)
        datasets_y = jnp.concatenate([datasets_y, new_datasets_y], axis=1)
        datasets_x, datasets_y = jax.vmap(sort_select)(datasets_x, datasets_y)
        
        return key, datasets_x, datasets_y, histories

    def solve(
        self,
        objective_fn: Callable[[jax.Array], jax.Array],
        dim: int,
        seed: int = 0
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Solve the optimization problem.
        
        Args:
            objective_fn: The function to minimize. Input shape (num_parallel, batch_size, dim), 
                          output shape (num_parallel, batch_size).
            dim: Dimension of the input x.
            seed: Random seed.
            
        Returns:
            Best x found and its corresponding y value.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        key = jax.random.PRNGKey(seed)
        
        # Initial sampling
        key, subkey = jax.random.split(key)
        datasets_x = jax.vmap(partial(latin_hyper_cube, num=self.batch_size, dim=dim))(
            jax.random.split(subkey, num=self.num_parallel)
        )
        datasets_y = objective_fn(datasets_x)
        
        histories = None
        mins_y = jnp.min(datasets_y, axis=1)
        
        for step in range(1, self.max_iter + 1):
            if self.debug:
                print(f"[INFO]  Start step {step}/{self.max_iter}, current best value = {print_with_prefix(mins_y)}")
            
            key, datasets_x, datasets_y, histories = self.gm_step(
                objective_fn, key, datasets_x, datasets_y, histories
            )
            
            mins_y = jnp.min(datasets_y, axis=1)
            if self.debug:
                print(f"[INFO]  End step {step}/{self.max_iter}, current best value = {print_with_prefix(mins_y)}")
                
        # Return the best result across all parallels
        best_idx = jnp.argmin(mins_y)
        best_x_idx = jnp.argmin(datasets_y[best_idx])
        return datasets_x[best_idx, best_x_idx], datasets_y[best_idx, best_x_idx]

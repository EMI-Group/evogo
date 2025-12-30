from typing import Callable, Tuple

import numpy as np
import torch
from torch import optim
from torch import nn

from evogo_torch.main import ArgsProtocol

from evogo_torch.trainer import train_model
from evogo_torch.models import (
    data_split,
    denormalize,
    normalize_with,
    standardize_with,
    MaternGaussianProcess,
    MarginalLogLikelihood,
    GPEvaluator,
    GPPairDiffEvaluator,
    MLPPredictor,
    MultilaneMSELoss,
    VAELoss,
    GenerativeModel,
    PairedGenerativeLoss,
)
from evogo_torch.utils import latin_hyper_cube, sort_select


def evogo_step(
    ARGS: ArgsProtocol,
    eval_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    datasets_x: torch.Tensor,
    datasets_y: torch.Tensor,
    histories: Tuple[torch.Tensor, torch.Tensor] | None = None,
    DEBUG: bool = False,
    SAVE: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Apply the generative model for optimization to the given dataset for one step

    Args:
        `datasets_x`: the datasets of inputs with dimension [num_parallels, dataset_size, dim(x)]
        `datasets_y`: the datasets of outputs with dimension [num_parallels, dataset_size]
        `histories`: the datasets of inputs and outputs of last iteration, can be None

    Returns:
        `datasets_x`, `datasets_y` and `histories` of next step
    """

    # get GP dataset
    (
        (all_datasets_x, all_datasets_y),
        (wins_x, loses_x, wins_y, loses_y),
        (de_norm_x, de_std_y),
    ) = data_split(
        datasets_x,
        datasets_y,
        histories=histories,
        portion=ARGS.portion,
        sliding_window=ARGS.slide_window,
    )
    num_parallels = datasets_x.size(0)
    dataset_size = all_datasets_x.size(1)
    dim = datasets_x.size(-1)
    device = datasets_x.device

    # train GP
    if ARGS.use_gp:
        lr = max(dataset_size / 600, 0.1) * 0.01 #* dataset_size
        gp_net = MaternGaussianProcess(dim, num_parallels)
        gp_net = MarginalLogLikelihood(gp_net)
        tx = optim.Adam(gp_net.parameters(), lr=lr, fused=True)
        print(f"[INFO]  Training Gaussian Process with learning rate = {lr}...")
        best_gp_params, trained = train_model(
            batch_size=dataset_size,
            epochs=1600,
            net=gp_net,
            tx=tx,
            shuffle=False,
            DEBUG=DEBUG,
            valid_portion=0,
            loss_names={},
            all_datasets_x=all_datasets_x,
            all_datasets_y=all_datasets_y,
        )
        if not trained.all():
            print("[ERROR] Not trained GP detected, skip current iteration")
            return datasets_x, datasets_y, histories
        # prepare for training generative model
        with torch.no_grad():
            gp_net.load_state_dict(best_gp_params)
            gp_net = gp_net.gp
            gp_net.eval()
            eval_diff_fn = GPPairDiffEvaluator(gp_net, all_datasets_x, all_datasets_y)
            eval_single_fn = GPEvaluator(gp_net, all_datasets_x, all_datasets_y)

    # train MLP if indicated
    elif ARGS.use_mlp:
        lr = 0.025 / all_datasets_x.size(1) #* min(64, all_datasets_x.size(1))
        mlp_net = MLPPredictor(dim, num_parallels)
        mlp_net = MultilaneMSELoss(model=mlp_net)
        tx = optim.Adam(mlp_net.parameters(), lr=lr, fused=True)
        print(f"[INFO]  Training MLP Predictor with learning rate = {lr}...")
        best_mlp_params, trained = train_model(
            batch_size=min(64, all_datasets_x.size(1)),
            epochs=200,
            net=mlp_net,
            tx=tx,
            shuffle=True,
            DEBUG=DEBUG,
            valid_portion=0.1,
            loss_names={},
            all_datasets_x=all_datasets_x,
            all_datasets_y=all_datasets_y,
        )
        if not trained.all():
            print("[ERROR] Not trained MLP detected, skip current iteration")
            return datasets_x, datasets_y, histories

        class _MLPEval(nn.Module):
            def __init__(self, mlp_net: MLPPredictor):
                super().__init__()
                self.mlp = mlp_net

        class _MLPEvalDiff(_MLPEval):
            def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor):
                out1 = self.mlp.forward(inputs1)
                out2 = self.mlp.forward(inputs2)
                return out1 - out2, torch.ones_like(out2)

        class _MLPEvalSingle(_MLPEval):
            def forward(self, inputs: torch.Tensor):
                out = self.mlp.forward(inputs)
                return out, torch.ones_like(out)

        with torch.no_grad():
            mlp_net.load_state_dict(best_mlp_params)
            mlp_net = mlp_net.model
            mlp_net.eval()
            eval_diff_fn = _MLPEvalDiff(mlp_net)
            eval_single_fn = _MLPEvalSingle(mlp_net)

    # use true evaluation if indicated
    elif ARGS.use_direct:

        class _TrueEval(nn.Module):
            def __init__(
                self,
                fn: nn.Module,
                de_norm_x: Tuple[torch.Tensor, torch.Tensor],
                de_std_y: Tuple[torch.Tensor, torch.Tensor],
            ):
                super().__init__()
                self.fn = fn
                self.de_norm_x_max = de_norm_x[0]
                self.de_norm_x_min = de_norm_x[1]
                self.de_std_y_mean = de_std_y[0]
                self.de_std_y_std = de_std_y[1]

        class _TrueEvalDiff(_TrueEval):
            def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor):
                inputs1 = denormalize(inputs1, self.de_norm_x_max, self.de_norm_x_min)
                inputs2 = denormalize(inputs2, self.de_norm_x_max, self.de_norm_x_min)
                out1 = self.fn.forward(inputs1)
                out2 = self.fn.forward(inputs2)
                out1 = standardize_with(out1, self.de_std_y_mean, self.de_std_y_std)
                out2 = standardize_with(out2, self.de_std_y_mean, self.de_std_y_std)
                return out1 - out2, torch.ones_like(out2)

        class _TrueEvalSingle(_TrueEval):
            def forward(self, inputs: torch.Tensor):
                inputs = denormalize(inputs, self.de_norm_x_max, self.de_norm_x_min)
                out = self.fn.forward(inputs)
                out = standardize_with(out, self.de_std_y_mean, self.de_std_y_std)
                return out, torch.ones_like(out)

        with torch.no_grad():
            eval_diff_fn = _TrueEvalDiff(eval_fn, de_norm_x, de_std_y)
            eval_single_fn = _TrueEvalSingle(eval_fn, de_norm_x, de_std_y)

        print("[INFO]  Using direct function...")
    else:
        eval_diff_fn = None
        raise NotImplementedError("Unexpected predictor type")

    # sample new points if necessary
    if ARGS.gm_batch_size > datasets_x.size(1):
        if not ARGS.sample_via_model:
            # using x heuristic
            temp_datasets_x = latin_hyper_cube(
                batch_size=num_parallels, num=ARGS.gm_batch_size * 10, dim=dim, device=device
            )
            multiple_to_original = ARGS.gm_batch_size // datasets_x.size(1)
            take_fn = torch.vmap(lambda x, p: x[p])
            set_fn = torch.vmap(lambda x, p, v: torch.index_fill(x, 0, p, v), in_dims=(0, 0, None))
            pair_diff_x = torch.cdist(temp_datasets_x, normalize_with(datasets_x, *de_norm_x))
            index_all = torch.zeros(
                (num_parallels, datasets_x.size(1), multiple_to_original), dtype=torch.long, device=device
            )
            for i in range(multiple_to_original):
                _min_idx = torch.argmin(pair_diff_x, dim=1)
                pair_diff_x = set_fn(pair_diff_x, _min_idx, 1e2)
                index_all[:, :, i] = _min_idx
            temp_datasets_x = take_fn(temp_datasets_x, index_all.reshape(num_parallels, -1))
        else:
            # VAE
            lr = 4e-4 #* min(int(datasets_x.size(1) * 0.9), 64)
            vae_net = VAELoss(dim=dim, num_parallels=num_parallels, drop_rate=ARGS.drop_rate)
            tx = optim.Adam(vae_net.parameters(), lr=lr, fused=True)
            print(f"[INFO]  Training VAE with learning rate = {lr}...")
            best_vae_params, trained = train_model(
                batch_size=min(int(datasets_x.size(1) * 0.9), 64),
                epochs=200,
                valid_portion=0.1,
                net=vae_net,
                tx=tx,
                shuffle=True,
                DEBUG=DEBUG,
                loss_names={"reconstruct": 0, "KL divergence": 1},
                real_xs=normalize_with(datasets_x, *de_norm_x),
            )
            with torch.no_grad():
                vae_net.load_state_dict(best_vae_params)
                vae_net.eval()
                vae_decoder_net = vae_net.decoder
                fake_dim = vae_net.encode_size
                temp_datasets_x: torch.Tensor = vae_decoder_net.forward(
                    torch.randn(num_parallels, ARGS.gm_batch_size, fake_dim, device=device)
                )

        # split via surrogate function
        # def eval_pair(in1: torch.Tensor):
        #     return eval_diff_fn.forward(
        #         torch.stack([in1] * datasets_x.size(1), dim=1),
        #         normalize_with(datasets_x, *de_norm_x),
        #     )

        # eval_pair = torch.vmap(eval_pair, in_dims=1, out_dims=(1, 1))
        # def get_temp_ys(temp_xs: torch.Tensor) -> torch.Tensor:
        #     pair_diff_y, _ = jax.jit(eval_pair)(temp_xs)
        #     pair_diff_y = pair_diff_y * de_std_y[1].unsqueeze(1).unsqueeze(1)
        #     temp_ys = pair_diff_y + datasets_y.unsqueeze(1)
        #     temp_ys = torch.mean(temp_ys, dim=-1)
        #     return temp_ys
        with torch.no_grad():
            temp_datasets_y = eval_single_fn.forward(temp_datasets_x)[0]
            _, (wins_x, loses_x, wins_y, loses_y), _ = data_split(temp_datasets_x, temp_datasets_y)
    # END sample new points if necessary

    # train generative model
    gm_net = GenerativeModel(dim, num_parallels, drop_rate=ARGS.drop_rate)
    pgl_net = PairedGenerativeLoss(
        eval_diff_fn,
        gm_net,
        cycle_scale=ARGS.cycle_scale,
        out_scale=1,
        mll_scale=1 if ARGS.use_fast_dist else 0.25,
        mip_scale=0.1,
        mip_std_scale=1,
        gan=ARGS.use_gan,
        single_gen=ARGS.use_single_gen,
        lcb=ARGS.use_lcb,
        discriminator=MLPPredictor(dim, num_parallels, drop_rate=ARGS.drop_rate) if ARGS.use_gan else None,
    )
    lr = (1500 / min(ARGS.gm_batch_size, 1000)) * 1e-5 #* ARGS.gm_batch_size
    epochs = 200
    if not ARGS.use_gp:
        lr = 1e-4 #* ARGS.gm_batch_size
    tx = optim.Adam(pgl_net.parameters(), lr=lr, fused=True)
    print(f"[INFO]  Training generative model with batch size = {ARGS.gm_batch_size}, learning rate = {lr} ...")
    best_params, trained = train_model(
        batch_size=ARGS.gm_batch_size,
        epochs=int(epochs * max(1000 / max(datasets_x.size(1), ARGS.gm_batch_size), 0.25)),
        net=pgl_net,
        tx=tx,
        shuffle=True,
        DEBUG=DEBUG,
        valid_portion=0.0,
        loss_names={
            "lose cycle": 0,
            "win cycle": 1,
            "lose out": 2,
            "win out": 3,
            "win - lose": 4,
            "lose - win": 5,
            "MIP": 6,
            "MIP 2": -1,
        },
        wins_x=wins_x,
        loses_x=loses_x,
        wins_y=wins_y,
        loses_y=loses_y,
    )
    if not trained.all():
        print("[ERROR] Not trained generative model detected, skip current iteration")
        return datasets_x, datasets_y, histories
    # get next step data
    with torch.no_grad():
        pgl_net.load_state_dict(best_params)
        gm_net = pgl_net.lose2win
        gm_net.eval()
        histories = (datasets_x, datasets_y)
        new_datasets_x = gm_net.forward(normalize_with(datasets_x, *de_norm_x))

        if SAVE is not None:
            if ARGS.save_count > 0:
                save_org_x = latin_hyper_cube(ARGS.save_count, datasets_x.size(-1))
                save_org_x = save_org_x.reshape(1, *save_org_x.shape)
                save_new_x = gm_net.forward(save_org_x)
                for i, (sox, snx) in enumerate(zip(save_org_x, save_new_x)):
                    sx = torch.cat([sox, snx], dim=1).cpu().numpy()
                    file = f"{ARGS.out}/save_func{ARGS.func_id}_batch{ARGS.save_count}_{i}.csv"
                    with open(file, "a") as f:
                        np.savetxt(f, np.asarray(sx, dtype=np.float32), delimiter=",")
            else:
                save_new_x = denormalize(new_datasets_x, *de_norm_x)
                for i, sx in enumerate(save_new_x):
                    file = f"{ARGS.out}/save_func{ARGS.func_id}_batch{ARGS.save_count}_{i}.csv"
                    with open(file, "a") as f:
                        np.savetxt(f, np.asarray(sx.cpu().numpy(), dtype=np.float32), delimiter=",")

        new_datasets_x = torch.clip(new_datasets_x, min=-0.25, max=1.25)
        new_datasets_x = denormalize(new_datasets_x, *de_norm_x)
        new_datasets_x = torch.clip(new_datasets_x, min=0, max=1)
        if eval_fn is not None:
            new_datasets_y = eval_fn(new_datasets_x)
            datasets_x = torch.concatenate([datasets_x, new_datasets_x], dim=1)
            datasets_y = torch.concatenate([datasets_y, new_datasets_y], dim=1)
            datasets_x, datasets_y = sort_select(datasets_x, datasets_y)
        else:
            datasets_x = new_datasets_x
        return datasets_x, datasets_y, histories

import time
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import torch
from torch import nn

from evogo_torch.utils import print_with_prefix


def prepare_epoch(batch_size: int, datasets: List[torch.Tensor], shuffle: bool = True) -> List[torch.Tensor]:
    """
    Prepare a (shuffled) epoch with given batch size over given datasets

    Args:
        `batch_size`: the batch size
        `key`: the random key
        `datasets`: the datasets, each one has dimension [num_parallels, dataset_size, *dim]
        `shuffle`: whether shuffle or not

    Returns:
        The datasets with each one splitted to [num_parallels, num_batches, batch_size, *dim]
    """
    num_parallels = datasets[0].shape[0]
    dataset_size = datasets[0].shape[1]
    num_batches = dataset_size // batch_size
    device = datasets[0].device
    if shuffle:
        perms = [torch.randperm(dataset_size, device=device) for _ in range(num_parallels)]
        perms = [
            p[: num_batches * batch_size].reshape(num_batches, batch_size) for p in perms
        ]  # skip incomplete batch
        device = datasets[0].device
        perms = [
            torch.arange(num_parallels, device=device).view([num_parallels] + [1] * perms[0].ndim),
            torch.stack(perms),
        ]
        datasets = [dataset.__getitem__(perms).swapaxes(0, 1) for dataset in datasets]
        # datasets = [
        #     torch.stack(
        #         [
        #             torch.stack(
        #                 [
        #                     data[p]
        #                     for p in pb  # p: [batch_size]
        #                 ]  # loop over batches, out: [num_batches, batch_size, *dim]
        #             )
        #             for data, pb in zip(
        #                 dataset, perms
        #             )  # data: [dataset_size, *dim], pb: [num_batches, batch_size]
        #         ],  # loop over parallels, out: [num_parallels, num_batches, batch_size, *dim]
        #         dim=1,  # out: [num_batches, num_parallels, batch_size, *dim]
        #     )
        #     for dataset in datasets  # loop over datasets
        # ]
    else:
        datasets = [torch.stack(torch.split(dataset, num_batches, dim=1)).swapaxes(0, 2) for dataset in datasets]
    return datasets


def train_valid_split(
    org_datasets: Dict[str, torch.Tensor], valid_portion: float = 0.1
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Split train and validation sets from given datasets

    Args:
        `org_datasets`: the datasets, each one has dimension [num_parallels, dataset_size, *dim]
        `valid_portion`: the validation set size portion

    Returns:
        `(datasets, validsets)`: the output training and validation sets
    """
    print(
        f"[DEBUG] Splitting datasets {tuple(org_datasets.keys())} of size {tuple(a.shape for a in org_datasets.values())} with validation portion = {valid_portion}"
    )
    datasets: List[torch.Tensor] = list(org_datasets.values())
    device = datasets[0].device
    num_parallels = datasets[0].shape[0]
    dataset_size = datasets[0].shape[1]
    num_valid = math.floor(dataset_size * valid_portion)
    perms = [torch.randperm(dataset_size, device=device) for _ in range(num_parallels)]
    valid_perms = [p[:num_valid] for p in perms]
    data_perms = [p[num_valid:] for p in perms]
    validsets = [torch.stack([d[p] for d, p in zip(dataset, valid_perms)]) for dataset in datasets]
    datasets = [torch.stack([d[p] for d, p in zip(dataset, data_perms)]) for dataset in datasets]
    return datasets, validsets


def train_epoch(
    batch_size: int,
    datasets: List[torch.Tensor],
    loss_fn: nn.Module,
    loss_shape: int | Sequence[int],
    tx: torch.optim.Optimizer,
    shuffle: bool = True,
) -> torch.Tensor:
    """
    Train a epoch with prepared epoch batches

    Args:
        `batch_size`: the batch size
        `datasets`: the datasets, each one has dimension [num_parallels, dataset_size, *dim]
        `loss_fn`: the loss value function with inputs (*batched_inputs)
        `loss_shape`: the shape of loss values
        `tx`: the optimizer
        `shuffle`: whether shuffle datasets or not

    Returns:
        `loss_val`: the averaged loss value of this epoch
    """
    batches_all = prepare_epoch(batch_size, datasets, shuffle=shuffle)
    num_batches = batches_all[0].shape[0]
    device = batches_all[0].device
    loss_val_all = torch.zeros(loss_shape, device=device)

    for batch_inputs in zip(*batches_all):
        tx.zero_grad()
        loss_vals: torch.Tensor = loss_fn(*batch_inputs)
        loss = loss_vals.sum()
        loss.backward()
        tx.step()
        loss_val_all += loss_vals.detach()

    return loss_val_all / num_batches


ParamsType = Dict[str, torch.Tensor]


def _clone_params(params: ParamsType):
    return {k: v.detach().clone() for k, v in params.items()}


def _get_parallel_best_params(
    loss_vals: torch.Tensor, params: ParamsType, prev_loss: torch.Tensor, prev_params: ParamsType
) -> Tuple[torch.Tensor, ParamsType]:
    replace_mask = loss_vals <= prev_loss
    for k, nv in params.items():
        pv = prev_params[k]
        pv = torch.where(replace_mask.view([replace_mask.size(0)] + [1] * (nv.ndim - 1)), nv.detach(), pv, out=pv)
    return torch.where(replace_mask, loss_vals, prev_loss), prev_params


_get_parallel_best_params = torch.compile(_get_parallel_best_params)

COMPILE = True
FIRST = {}


_COLORS = ["b", "g", "r", "c", "m", "y", "k"]


def train_model(
    batch_size: int,
    epochs: int,
    net: nn.Module,
    tx: torch.optim.Optimizer,
    loss_names: Dict[str, int],
    shuffle: bool = True,
    valid_portion: float = 0.0,
    DEBUG: bool = False,
    **org_datasets: torch.Tensor,
):
    """
    Train the given `net` with initial `params` for `epochs` with given `datasets`

    Args:
        batch_size: the batch size
        epochs: the number of epochs
        net: the network
        lr: the learning rate
        loss_names: the loss names
        shuffle: whether shuffle or not
        valid_portion: the validation set size portion
        DEBUG: whether print debug info or not
        **org_datasets: the original datasets as inputs in the form of a dict

    Returns:
        The best parameters
    """
    _compiled_train_epoch = torch.compile(train_epoch, disable=not COMPILE)
    num_params = sum(p.numel() for p in net.parameters())
    _compile_str = "Compile" if COMPILE else "Not compile"
    print(f"[INFO]  {_compile_str} network {net.__class__.__name__} with {num_params:,} parameters")

    net.train()
    datasets = list(org_datasets.values())
    t0 = t_prev = time.time()
    if len(loss_names) > 0:
        prefix_cnt = max(max(len(n) for n, i in loss_names.items() if i >= 0), len("history")) + 15
    else:
        prefix_cnt = 0
    num_parallels = datasets[0].shape[0]
    num_losses = len(loss_names)
    loss_shape = (num_losses, num_parallels) if num_losses > 1 else (num_parallels,)
    best_params = (
        torch.full((num_parallels,), fill_value=1e10, device=datasets[0].device),
        _clone_params(dict(net.named_parameters())),
    )
    loss_vals_all: torch.Tensor = torch.zeros(epochs, num_parallels, device=datasets[0].device)
    if valid_portion > 0:  # with validation set
        datasets, validsets = train_valid_split(org_datasets, valid_portion=valid_portion)
    i_prev = 0
    for i in range(1, epochs + 1):
        # run epoch
        # global FIRST
        # if num_params not in FIRST:
        #     loss_vals = _compiled_train_epoch(batch_size, datasets, net, loss_shape, tx, shuffle=shuffle)
        #     FIRST[num_params] = False
        # else:
        #     with torch.compiler.set_stance("fail_on_recompile"):
        loss_vals = _compiled_train_epoch(batch_size, datasets, net, loss_shape, tx, shuffle=shuffle)
        # deal with loss
        if valid_portion > 0:  # with validation set
            loss_vals: torch.Tensor = net.forward(*validsets)
        total_loss = torch.sum(loss_vals, dim=0) if loss_vals.ndim > 1 else loss_vals
        loss_vals_all[i - 1] = total_loss
        # preserve best parameters
        if i > 1:
            best_params = _get_parallel_best_params(
                total_loss, dict(net.named_parameters()), best_params[0], best_params[1]
            )
        # prints
        if i % max(epochs // 4, 1) == 0 or time.time() - t_prev >= 600 or (DEBUG and i % max(epochs // 20, 1) == 0):
            print(
                f"[INFO]  Loss epoch {i}/{epochs}, total_loss = {print_with_prefix(total_loss)}, elapsed = {time.time() - t0}:"
            )
            if len(loss_names) <= 0:
                t_prev = time.time()
                i_prev = i
                continue
            print(
                " " * (prefix_cnt - len("history") - 3)
                + f"history = {print_with_prefix(loss_vals_all[i_prev : i - 1], prefix_cnt=prefix_cnt)}"
            )
            for name, idx in loss_names.items():
                if idx < 0:
                    continue
                print(
                    " " * (prefix_cnt - len(name) - 3)
                    + f"{name} = {print_with_prefix(loss_vals[idx], prefix_cnt=prefix_cnt)}"
                )
            t_prev = time.time()
            i_prev = i
    # output
    print(f"[INFO]  Best total_loss = {print_with_prefix(best_params[0])}, elapsed = {time.time() - t0}")
    if DEBUG and HAS_PLT:
        plt.ion()
        plt.clf()
        loss_vals_all = np.transpose(loss_vals_all.detach().cpu().numpy())
        for i in range(num_parallels):
            plt.plot(loss_vals_all[i], _COLORS[i % len(_COLORS)])
        plt.show()
        plt.pause(1)
    return best_params[1], best_params[0] != 1e10

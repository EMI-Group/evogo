from typing import Any, Tuple
import numpy as np
import torch


def latin_hyper_cube(batch_size: int, n: int, d: int, device: torch.device | None = None) -> torch.Tensor:
    perms = torch.rand(batch_size, d, n, device=device)
    perms = torch.argsort(perms, dim=-1)
    reals = torch.rand(batch_size, d, n, device=device)
    samples = (reals + perms) / n
    return samples.swapaxes(-1, -2)


def sort_select(
    xs: torch.Tensor, ys: torch.Tensor, num: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num is None:
        num = xs.size(-2) // 2
    perm = torch.argsort(ys, dim=-1)
    if ys.ndim == 1:
        perm = perm[:num]
        return xs[perm], ys[perm]
    else:
        perm = perm.narrow(-1, 0, num)
        indices = [torch.arange(s, device=xs.device).view([1] * i + [s] + [1] * (xs.ndim - i - 2)) for i, s in enumerate(xs.size()[:-2])]
        indices.append(perm)
        return xs.__getitem__(indices), ys.gather(-1, perm)


def print_with_prefix(array: Any, prefix_cnt: int | None = None) -> str:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    else:
        array = np.asarray(array)
    if prefix_cnt is not None:
        return np.array2string(array, prefix=" " * prefix_cnt, max_line_width=1000)
    else:
        return np.array2string(array, max_line_width=1000)

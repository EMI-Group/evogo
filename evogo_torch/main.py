import argparse
import tomli
import sys
import os
import signal
import subprocess
from typing import NamedTuple, Union, List
import io
from contextlib import redirect_stdout, redirect_stderr
import warnings

import numpy
if "bool8" not in numpy.__dict__:
    numpy.bool8 = numpy.bool_  # For compatibility with numpy < 1.26

import torch

from evogo_torch import models
from evogo_torch import trainer
from evogo_torch.utils import latin_hyper_cube, print_with_prefix
from evogo_torch.function_defs import get_functions
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

def _suppress_io():
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    cm_warn = warnings.catch_warnings()
    cm_warn.__enter__()
    warnings.simplefilter("ignore")
    cm_out = redirect_stdout(buf_out)
    cm_err = redirect_stderr(buf_err)
    cm_out.__enter__()
    cm_err.__enter__()
    return buf_out, buf_err, cm_out, cm_err, cm_warn

def _restore_io(cm):
    buf_out, buf_err, cm_out, cm_err, cm_warn = cm
    try:
        cm_err.__exit__(None, None, None)
    finally:
        cm_out.__exit__(None, None, None)
        cm_warn.__exit__(None, None, None)


def signal_handler(signal, frame):
    print("\n[INFO]  Caught Ctrl+C / SIGINT signal")
    s = input("Type 'yes' to quit or 'no' (default) to continue\n")
    if s.lower() == "yes":
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def str2bool(v: str) -> bool:
    """Convert string to boolean with flexible input support."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_list(value: str) -> Union[List[int], int, None]:
    """Parse list from string input."""
    try:
        return eval(value)
    except:  # noqa: E722
        return value


def load_config() -> dict:
    """Load default configuration from TOML file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.toml")
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        print(f"[WARNING] Config file not found at {config_path}, using built-in defaults")
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return {}


class ArgsProtocol(NamedTuple):
    # Hardware configuration
    gpu_id: int
    out: str
    # Visualization parameters
    save_iter: List[int] | int | None
    save_count: int
    # Batch parameters
    max_iter: int
    func_id: int
    num_parallel: int
    repeats: int
    force_repeat: int
    batch_size: int
    gm_batch_size: int
    # Model configuration
    use_inv: bool
    sample_via_model: bool
    use_fast_dist: bool
    compile: bool
    # Ablation parameters
    use_gp: bool
    use_mlp: bool
    use_direct: bool
    use_gan: bool
    use_single_gen: bool
    use_lcb: bool
    # Parameters
    portion: float
    slide_window: float
    cycle_scale: float
    drop_rate: float


def setup_argparse(defaults: dict) -> ArgsProtocol:
    """Set up command-line arguments with TOML defaults."""
    parser = argparse.ArgumentParser(
        description="Optimization Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
        exit_on_error=False,
    )

    # Hardware configuration
    parser.add_argument(
        "--out",
        type=str,
        default=defaults.get("out", "results"),
        help="The output folder",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=defaults.get("gpu_id", 0),
        help="GPU device ID to use",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=defaults.get("max_iter", 10),
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--func-id",
        type=int,
        default=defaults.get("func_id", 0),
        help="Function ID for optimization",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=defaults.get("num_parallel", 10),
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=defaults.get("repeats", 1),
        help="Number of experiment repeats",
    )
    parser.add_argument(
        "--force-repeat",
        type=int,
        default=defaults.get("force_repeat", -1),
        help="Force specific repeat index (-1 to disable)",
    )

    # Visualization parameters
    parser.add_argument(
        "--save-iter",
        type=parse_list,
        default=defaults.get("save_iter", list(range(10))),
        help="Iterations to save results",
    )
    parser.add_argument(
        "--save-count",
        type=int,
        default=defaults.get("save_count", -1),
        help="Number of saves to keep (-1 for unlimited)",
    )

    # Batch parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.get("batch_size", 100),
        help="Main batch size for processing",
    )
    parser.add_argument(
        "--gm-batch-size",
        type=int,
        default=defaults.get("gm_batch_size", 1000),
        help="Generator batch size",
    )

    # Model configuration
    parser.add_argument(
        "--use-inv",
        type=str2bool,
        default=defaults.get("use_inv", True),
        help="Enable inverse transformations",
    )
    parser.add_argument(
        "--sample-via-model",
        type=str2bool,
        default=defaults.get("sample-via-model", True),
        help="Use VAE model for data augmentation or heuristic",
    )
    parser.add_argument(
        "--use-fast-dist",
        type=str2bool,
        default=defaults.get("use_fast_dist", True),
        help="Enable fast distance calculations",
    )
    parser.add_argument(
        "--compile",
        type=str2bool,
        default=defaults.get("compile", False),
        help="Enable training compilation",
    )

    # Ablation parameters
    parser.add_argument(
        "--use-gp",
        type=str2bool,
        default=defaults.get("use_gp", False),
        help="Enable GP components",
    )
    parser.add_argument(
        "--use-mlp",
        type=str2bool,
        default=defaults.get("use_mlp", False),
        help="Enable MLP components",
    )
    parser.add_argument(
        "--use-direct",
        type=str2bool,
        default=defaults.get("use_direct", False),
        help="Enable real function components",
    )
    parser.add_argument(
        "--use-gan",
        type=str2bool,
        default=defaults.get("use_gan", False),
        help="Use GAN model instead of evogo",
    )
    parser.add_argument(
        "--use-single-gen",
        type=str2bool,
        default=defaults.get("use_single_gen", False),
        help="Use single generative model instead of paired ones",
    )
    parser.add_argument(
        "--use-lcb",
        type=str2bool,
        default=defaults.get("use_lcb", False),
        help="Use LCB loss instead of KG loss",
    )

    # Parameters
    parser.add_argument(
        "--portion",
        type=float,
        default=defaults.get("portion", 0.1),
        help="Data portion for sampling",
    )
    parser.add_argument(
        "--cycle-scale",
        type=float,
        default=defaults.get("cycle_scale", 400.0),
        help="Scaling factor for cycles",
    )
    parser.add_argument(
        "--slide-window",
        type=float,
        default=defaults.get("slide_window", 0.3),
        help="The size of the sliding window for surrogate training",
    )
    parser.add_argument(
        "--drop-rate",
        type=float,
        default=defaults.get("drop_rate", 1 / 128),
        help="The dropout rate for generative and predictive model training",
    )

    return parser.parse_args()


def configure_runtime(args: ArgsProtocol):
    """Configure runtime parameters based on arguments."""
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # Set global parameters
    models.USE_INV = args.use_inv
    models.USE_FAST_DIST = args.use_fast_dist
    trainer.COMPILE = args.compile


def write_min_to_file(file: str, ys: torch.Tensor):
    with open(file, "a") as f:
        mins = torch.min(ys, dim=1).values
        for m in mins:
            f.write(f"{m.item()}\t")
        f.write("\n")
    return mins


def run_optimization(ARGS: ArgsProtocol, debug: bool = False, idx_repeat: int = 0):
    """Main optimization loop."""

    # Load functions
    torch.set_default_device("cuda:0")
    functions, dimensions = get_functions(
        seed=ARGS.func_id * 1000,
        device=torch.device("cuda:0"),
        instances=ARGS.num_parallel,
    )

    # File handling and initialization
    dim = dimensions[ARGS.func_id]
    os.makedirs(ARGS.out, exist_ok=True)
    file_path = (
        f"results_func{ARGS.func_id}_dim{dim}_predict{ARGS.use_mlp:d}{ARGS.use_direct:d}{ARGS.use_gp:d}"
        + f"_gan{ARGS.use_gan}_single{ARGS.use_single_gen}"
        + f"_portion{ARGS.portion}_slide{ARGS.slide_window}_cyc{ARGS.cycle_scale:0.0f}"
    )
    file_path = os.path.join(ARGS.out, file_path)

    with open(file_path, "w") as f:
        f.write("")

    # Initialization
    seed = idx_repeat * 10000 + ARGS.func_id * 1000 + ARGS.batch_size // 10 + ARGS.gpu_id
    torch.manual_seed(seed)
    print(
        f"[START] func_id={ARGS.func_id}, dim={dim}, batch={ARGS.batch_size}/{ARGS.gm_batch_size}, GPU={ARGS.gpu_id}"
    )

    # Main optimization loop
    
    # import jax
    # key = jax.random.PRNGKey(seed)
    # from evogo.utils import latin_hyper_cube as jax_latin_hyper_cube
    # from evogo.function_defs import get_functions as jax_get_functions
    # jax_functions, _ = jax_get_functions(ARGS.func_id, ARGS.num_parallel)
    # datasets_x = jax.vmap(jax_latin_hyper_cube, in_axes=(0, None, None))(jax.random.split(key, ARGS.num_parallel), ARGS.batch_size, dim)
    # datasets_y = jax_functions[ARGS.func_id](datasets_x)
    # datasets_x = torch.from_dlpack(jax.dlpack.to_dlpack(datasets_x)).to(torch.float32)
    # datasets_y = torch.from_dlpack(jax.dlpack.to_dlpack(datasets_y)).to(torch.float32)
    
    datasets_x = latin_hyper_cube(ARGS.num_parallel, ARGS.batch_size, dim)
    if not debug:
        cm0 = _suppress_io()
    try:
        datasets_y = functions[ARGS.func_id](datasets_x)
    finally:
        if not debug:
            _restore_io(cm0)
    mins_y = write_min_to_file(file_path, datasets_y)

    histories = None

    from evogo_torch.step import evogo_step

    for step in range(1, ARGS.max_iter + 1):
        # original verbose logging style
        print(f"[INFO]  [ITER {step}/{ARGS.max_iter}] Best: {print_with_prefix(mins_y)}")
        if not debug:
            cm1 = _suppress_io()
        try:
            datasets_x, datasets_y, histories = evogo_step(
                ARGS,
                functions[ARGS.func_id],
                datasets_x,
                datasets_y,
                histories,
                debug,
                (
                    step
                    if (step == ARGS.save_iter or (isinstance(ARGS.save_iter, list) and step in ARGS.save_iter))
                    else None
                ),
            )
        finally:
            if not debug:
                _restore_io(cm1)
        mins_y = write_min_to_file(file_path, datasets_y)
        print(f"[INFO]  [ITER {step}/{ARGS.max_iter}] New best: {print_with_prefix(mins_y)}")

def main(debug: bool = False, **config):
    # Load configuration
    config_defaults = load_config()
    
    # Override with environment variables
    for key in os.environ:
        if key.lower() in ["gpu_id", "out", "save_iter", "save_count", "max_iter", "func_id", 
                           "num_parallel", "repeats", "force_repeat", "batch_size", "gm_batch_size",
                           "use_inv", "sample_via_model", "use_fast_dist", "compile", 
                           "use_gp", "use_mlp", "use_direct", "use_gan", "use_single_gen", "use_lcb",
                           "portion", "cycle_scale", "slide_window", "drop_rate"]:
            val = os.environ[key]
            # Try to parse as int or float or bool or list
            try:
                if val.lower() in ("true", "false"):
                    val = str2bool(val)
                elif val.startswith("[") and val.endswith("]"):
                    val = parse_list(val)
                elif "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except:
                pass
            config_defaults[key.lower()] = val

    config_defaults.update(config)
    use_argparse = config_defaults.get("use_argparse", True)
    config_defaults.pop("use_argparse", None)
    gpu_id = config_defaults.get("gpu_id", 0)
    if use_argparse:
        try:
            args = setup_argparse(config_defaults)
            args = vars(args)
            args["gm_batch_size"] = max(int(args.get("gm_batch_size", 10)),
                                        int(args.get("batch_size", 10)))
            args = ArgsProtocol(**args)
        except Exception as e:
            print(f"[INFO]  Get arguments ended with error '{e}', using default")
            args = ArgsProtocol(**config_defaults)
    else:
        args = ArgsProtocol(**config_defaults)
    print(f"[INFO]  Execution arguments: {args}")
    configure_runtime(args)

    # Run optimization workflow
    if args.force_repeat >= 0:
        run_optimization(args, debug=debug, idx_repeat=args.force_repeat)
    else:
        for i in range(args.repeats):
            run_optimization(args, debug=debug, idx_repeat=i)


if __name__ == "__main__":
    os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
    torch.set_float32_matmul_precision('high')
    main(debug=False)

"""Vitis-AI quantization script for FPGA deployment.

Called exclusively via `docker exec` from deploy.py (running inside the vitis-ai-pytorch
container). Never imported directly — the subprocess / CLI boundary is intentional because
this script depends on pytorch_nndct, which is only available inside the container.

Usage (via deploy.py / docker exec):
    python DDC_FPGA/scripts/fpga/model_quant.py --run_dir <RUN_DIR> --quant_mode calib ...

Modes:
    float  — evaluate float model (optionally inspect for DPU compatibility)
    calib  — calibrate quantization (always run before test/deploy)
    test   — evaluate quantized model; with --deploy exports the .xmodel
"""

import argparse
import os
import random
import sys
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Optional, Tuple

import h5py
import torch
from omegaconf import OmegaConf
from pytorch_nndct.apis import torch_quantizer  # type: ignore
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

# ---- Project root setup (must happen before src imports) ----
project_root = Path(__file__).resolve().parent.parent.parent
os.environ["PROJECT_ROOT"] = str(project_root)
sys.path.append(str(project_root))
from src.models.components.dpu_wrapper import (  # noqa: E402
    FactorizedPriorDPUWrapper,
    ResidualScaleHyperpriorDPUWrapper,
)
from src.models.components.res_factorized_prior_dpu import (  # noqa: E402
    ResidualFactorizedPriorPatched,
)
from src.models.components.res_scale_hyperprior_dpu import (  # noqa: E402
    ResidualScaleHyperpriorPatched,
)
from src.utils.constants import AMP_MAX, AMP_MIN  # noqa: E402
from src.utils.metrics import MerlinRDLoss  # noqa: E402

# ============================================================
# ARG PARSING
# ============================================================


def parse_args() -> Namespace:
    """Parse and validate CLI arguments.

    All side-effects are contained here.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the Hydra run directory (must contain .hydra/config.yaml and checkpoints/last.ckpt).",
    )
    parser.add_argument("--config_file", default=None, help="Quantization configuration file.")
    parser.add_argument(
        "--subset_len",
        default=None,
        type=int,
        help="Number of samples for calibration/evaluation. Uses full dataset if not set.",
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Input data batch size (default: 1)."
    )
    parser.add_argument(
        "--quant_mode",
        default="calib",
        choices=["float", "calib", "test"],
        help="float: evaluate float model | calib: calibrate | test: evaluate quantized model.",
    )
    parser.add_argument(
        "--fast_finetune", action="store_true", help="Fast finetune model before calibration."
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Export .xmodel for deployment (requires --quant_mode test).",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Run DPU Inspector (requires --quant_mode float)."
    )
    parser.add_argument("--target", nargs="?", const="", help="Target DPU architecture string.")

    args, _ = parser.parse_known_args()

    # ---- Derived fields ----
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise RuntimeError(f"run_dir does not exist: {run_dir}")

    args.model_dir = run_dir / "checkpoints"
    config_path = run_dir / ".hydra" / "config.yaml"
    args.hydra_conf = OmegaConf.load(config_path)

    args.data_dir = Path(args.hydra_conf.data.get("hdf5_dir", None))
    if not args.data_dir or not args.data_dir.exists():
        raise RuntimeError(f"data_dir does not exist: {args.data_dir}")

    # ---- Constraint enforcement ----
    if args.quant_mode != "test" and args.deploy:
        warnings.warn("--deploy requires --quant_mode test. Setting deploy=False.")
        args.deploy = False
    if args.inspect and args.quant_mode == "float" and args.batch_size != 1:
        warnings.warn("--inspect requires batch_size=1. Enforcing it.")
        args.batch_size = 1
    if args.deploy and (args.batch_size != 1 or args.subset_len != 1):
        warnings.warn("--deploy requires batch_size=1 and subset_len=1. Enforcing them.")
        args.batch_size = 1
        args.subset_len = 1

    return args


# ============================================================
# DATA
# ============================================================


class CustomDataset(Dataset):
    """Loads SAR patches from an HDF5 file.

    Avoids the heavier TSXSSCDataset imports which aren't available in the container.
    """

    def __init__(self, hdf5_path: Path):
        super().__init__()
        self.hdf5_path = hdf5_path
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        with h5py.File(self.hdf5_path, "r") as f:
            self.num_patches = f["patches"].shape[0]
            self.attrs = dict(f.attrs)

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return self.num_patches

    def __getitem__(self, idx):
        """Get a patch by index, applying log-scale normalisation."""
        with h5py.File(self.hdf5_path, "r") as f:
            patch = torch.from_numpy(f["patches"][idx]).float()
            # 4-channel HDF5 [real, imag, ADAM-NOC, MERLIN] — keep only real/imag
            patch = patch[:, :, 0:2]
            patch = torch.square(patch)
            patch = torch.log(patch + 1e-2)
            patch = (patch - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)
            return patch.permute(2, 0, 1)  # C, H, W


def load_data(
    data_dir: Path, subset_len: Optional[int], batch_size: int, split: str = "val"
) -> DataLoader:
    """Build a DataLoader for the given split, optionally limiting to a random subset."""
    dataset = CustomDataset(data_dir / f"{split}.h5")
    if subset_len and subset_len <= len(dataset):
        dataset = Subset(dataset, random.sample(range(len(dataset)), subset_len))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================
# MODEL LOADING
# ============================================================


def load_model(
    run_dir: Path, hydra_conf, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Instantiate and load weights into (full_float_model, dpu_wrapper).

    Returns:
        full_model: ResidualScaleHyperpriorPatched or ResidualFactorizedPriorPatched
                    with loaded weights (used for generating split-graph intermediates).
        model:      Matching DPU wrapper (quantized).
    """
    model_path = run_dir / "checkpoints" / "last.ckpt"
    # We cannot use hydra.utils.instantiate() (no lightning in container).
    # Parse _target_ and instantiate directly.
    model_name = hydra_conf.model.net._target_.split(".")[-1]
    model_params = hydra_conf.model.net
    model_params.pop("_target_")

    if model_name == "ResidualScaleHyperpriorPatched":
        model_params["export_dpu"] = True
        print(f"Instantiating ResidualScaleHyperpriorPatched with params: {model_params}")
        full_model = ResidualScaleHyperpriorPatched(**model_params).to(device)

        checkpoint = torch.load(model_path)
        if "state_dict" in checkpoint:  # Lightning checkpoint format
            state_dict = {k.replace("net.", "", 1): v for k, v in checkpoint["state_dict"].items()}
            full_model.load_state_dict(state_dict, strict=False)
        else:
            full_model.load_state_dict(checkpoint, strict=False)

        # Disable gradients to avoid VAI_Q trace errors
        for param in full_model.parameters():
            param.requires_grad = False

        print("Wrapping in ResidualScaleHyperpriorDPUWrapper...")
        model = ResidualScaleHyperpriorDPUWrapper(full_model)
    elif model_name == "ResidualFactorizedPriorPatched":
        model_params["export_dpu"] = True
        print(f"Instantiating ResidualFactorizedPriorPatched with params: {model_params}")
        full_model = ResidualFactorizedPriorPatched(**model_params).to(device)

        checkpoint = torch.load(model_path)
        if "state_dict" in checkpoint:  # Lightning checkpoint format
            state_dict = {k.replace("net.", "", 1): v for k, v in checkpoint["state_dict"].items()}
            full_model.load_state_dict(state_dict, strict=False)
        else:
            full_model.load_state_dict(checkpoint, strict=False)

        for param in full_model.parameters():
            param.requires_grad = False

        print("Wrapping in FactorizedPriorDPUWrapper...")
        model = FactorizedPriorDPUWrapper(full_model)
    else:
        raise ValueError(f"Unrecognised model: {model_name}")

    print("Checkpoint loaded successfully.")
    return full_model, model


# ============================================================
# EVALUATION
# ============================================================


def evaluate_hyperprior(
    quant_model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    float_model: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the (quantized) ResidualScaleHyperprior model on the validation set.

    Uses the 4-input split-graph strategy: float_model generates
    (x_real, abs_y, z_hat, y_hat) that the quantized DPU wrapper expects.
    A meaningful loss cannot be computed in split mode, so 0.0 is always returned.
    """
    quant_model = quant_model.to(device)
    float_model = float_model.to(device)
    float_model.eval()

    nb_images = 0
    loss_total = 0.0

    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs = data.to(device).float()

            # --- Split Graph Evaluation Strategy ---
            # We need to feed 4 inputs to quant_model: (x, abs_y, z_hat, y_hat).
            # We generate valid intermediate maps using the float_model.

            # 1. Prepare inputs (matches logic in dpu_wrapper.py Mode 1)
            x_real = inputs[:, :1, :, :]
            x_imag = inputs[:, 1:, :, :]

            # 2. Run float model to get intermediates
            # We can't just run float_model(inputs) because we need the internals.
            y_real = float_model.g_a(x_real)
            y_imag = float_model.g_a(x_imag)
            y = torch.cat((y_real, y_imag), dim=1)

            # Intermediate 1: abs_y
            abs_y = torch.abs(y)

            # Intermediate 2: z_hat — in trace/calib mode we bypass entropy, so z_hat ≈ z
            z_hat = float_model.h_a(abs_y)

            # Intermediate 3: y_hat — in trace mode y_hat ≈ y
            y_hat = y

            # 3. Run quantized model with 4 inputs
            # For g_s input, take the real part (first half channels) as approximation.
            y_hat_effective = y_hat[:, : y_hat.shape[1] // 2, :, :]

            # For g_a input (first arg), we pass x_real (1 channel)
            _ = quant_model(x_real, abs_y, z_hat, y_hat_effective)

            # We cannot compute a meaningful loss in Split Mode (graph is broken).
            loss_total += 0.0
            nb_images += inputs.size(0)

    return loss_total / nb_images


def evaluate_factorized(
    quant_model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    float_model: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the (quantized) ResidualFactorizedPrior model on the validation set.

    Uses the 2-input split-graph strategy: float_model generates (x_real, y_real)
    that the quantized FactorizedPriorDPUWrapper expects.
    A meaningful loss cannot be computed in split mode, so 0.0 is always returned.
    """
    quant_model = quant_model.to(device)
    float_model = float_model.to(device)
    float_model.eval()

    nb_images = 0
    loss_total = 0.0

    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs = data.to(device).float()

            # Prepare single-channel slice (matches FactorizedPriorDPUWrapper Mode 2)
            x_real = inputs[:, :1, :, :]

            # Generate intermediate: g_a output as y passthrough (no entropy in trace)
            y_real = float_model.g_a(x_real)

            # Run quantized model with 2 inputs (split mode)
            _ = quant_model(x_real, y_real)

            loss_total += 0.0
            nb_images += inputs.size(0)

    return loss_total / nb_images


def calibrate(quantizer) -> None:
    """Export quantization config (calib mode only)."""
    quantizer.export_quant_config()


def export_xmodel(quantizer) -> None:
    """Export the compiled split .xmodel for Hybrid Inference on the DPU."""
    print("Exporting split xmodel for Hybrid Inference...")
    quantizer.export_xmodel(deploy_check=False)


def debug_forward_execution(model: torch.nn.Module, inputs: tuple, desc: str) -> None:
    """Trace which modules fire during a forward pass.

    Helps diagnose graph mismatches between calibration and deployment.
    """
    print(f"\n[DEBUG] --- Forward Execution Trace: {desc} ---")
    print(f"    [DEBUG] Input shapes: {[i.shape for i in inputs]}")

    hooks = []

    def get_hook(name):
        """Create a forward hook that prints module execution."""

        def hook(module, input, output):
            """Print only leaf modules or interesting ones to avoid spam."""
            if (
                len(list(module.children())) == 0
                or "Entropy" in module.__class__.__name__
                or "Gaussian" in module.__class__.__name__
            ):
                print(f"    [DEBUG] Executing: {name} (Type: {module.__class__.__name__})")

        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(get_hook(name)))

    try:
        with torch.no_grad():
            model(*inputs)
    except Exception as e:
        print(f"    [DEBUG] Forward pass crashed: {e}")
    finally:
        for h in hooks:
            h.remove()

    print("--------------------------------------------------\n")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        if k == "hydra_conf":
            continue  # skip — too verbose, use config_path for reference
        print(f"  {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model ----
    full_model, model = load_model(Path(args.run_dir), args.hydra_conf, device)
    model.eval()

    # ---- Dummy inputs and evaluate function (model-type-aware) ----
    H = 256
    ds = full_model.main_downsampling_factor  # 16 for both model families
    N = full_model.nb_channels_main
    M = 2 * N
    x_dumb = torch.randn(args.batch_size, 1, H, H).to(device)
    if isinstance(full_model, ResidualFactorizedPriorPatched):
        # FactorizedPrior: 2-input split (g_a | g_s)
        # y_hat: [B, N, H/ds, H/ds]
        y_hat_dumb = torch.randn(args.batch_size, N, H // ds, H // ds).to(device)
        dummy_inputs = (x_dumb, y_hat_dumb)
        eval_fn = evaluate_factorized
    else:
        # ResidualScaleHyperprior: 4-input split (g_a | h_a | h_s | g_s)
        # abs_y: [B, M, H/ds, H/ds]   z_hat: [B, M, H/ds/hyper_ds, H/ds/hyper_ds]   y_hat: [B, N, H/ds, H/ds]
        hyper_ds = full_model.hyper_downsampling_factor  # 8
        abs_y_dumb = torch.randn(args.batch_size, M, H // ds, H // ds).to(device)
        z_hat_dumb = torch.randn(args.batch_size, M, H // ds // hyper_ds, H // ds // hyper_ds).to(
            device
        )
        y_hat_dumb = torch.randn(args.batch_size, N, H // ds, H // ds).to(device)
        dummy_inputs = (x_dumb, abs_y_dumb, z_hat_dumb, y_hat_dumb)
        eval_fn = evaluate_hyperprior

    # ---- Float mode: evaluate or inspect ----
    if args.quant_mode == "float":
        quant_model = model
        if args.inspect:
            if not args.target:
                raise ValueError("--target must be specified for --inspect.")
            from pytorch_nndct.apis import Inspector  # type: ignore

            inspector = Inspector(args.target)
            inspector.inspect(model, dummy_inputs, device=device)
            return

    # ---- Quantized modes: calib / test ----
    else:
        # if args.quant_mode == "calib":
        #     # [DEBUG] Trace graph before calibration to catch graph mismatches early
        #     debug_forward_execution(
        #         model, dummy_inputs, f"Calibration Mode ({len(dummy_inputs)} inputs)"
        #     )

        quantizer = torch_quantizer(
            args.quant_mode,
            model,
            dummy_inputs,
            device=device,
            quant_config_file=args.config_file,
            target=args.target,
        )
        quant_model = quantizer.quant_model

    # ---- Loss function ----
    loss_params = args.hydra_conf.model.criterion
    loss_params.pop("_target_")
    loss_fn = MerlinRDLoss(**loss_params).to(device)

    # ---- Load data -----
    val_loader = load_data(args.data_dir, subset_len=args.subset_len, batch_size=args.batch_size)
    for data in val_loader:
        print(f"Data batch shape: {data.shape}")
        break

    # ---- Fast finetune ----
    if args.fast_finetune:
        ft_loader = load_data(
            args.data_dir, subset_len=50, batch_size=args.batch_size, split="train"
        )
        if args.quant_mode == "calib":
            quantizer.fast_finetune(eval_fn, (quant_model, ft_loader, loss_fn, full_model, device))
        elif args.quant_mode == "test":
            quantizer.load_ft_param()

    # ---- Evaluate ----
    print(f"Evaluating model in '{args.quant_mode}' mode...")
    loss_gen = eval_fn(quant_model, val_loader, loss_fn, full_model, device)
    print(f"Loss after evaluation: {loss_gen}")

    # ---- Export ----
    if args.quant_mode == "calib":
        calibrate(quantizer)
    if args.deploy:
        export_xmodel(quantizer)


if __name__ == "__main__":
    main()

"""
This script will be used to quantize a PyTorch model for FPGA deployment.
It is based on Vitis-AI resnet18 PyTorch model quantization example, available at: https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html#pytorch-tutorial

@TODO: update this docstring
What I will do in this script:
- Have 3 quantization mode: "float", perform no quantization: simply evaluates the float model, "calib" for calibration and "test" for evaluation.
- Have the necessary argparse:
    - quantization mode (float, calib, test)
    - data dir (we will manage automatically to select the right dataset based on the quantization mode)
    - model path
    - config_file
    - subset_len, to limit the number of samples used for calibration or testing
    - batch_size
    - fast_finetune, to perform fast finetuning before calibration
    - deploy, to export the xmodel for deployment
    - inspect
    - target, to specify the target device
"""

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import h5py
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_nndct.apis import torch_quantizer  # type: ignore
from torch.utils.data import Dataset
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
os.environ["PROJECT_ROOT"] = str(project_root)
sys.path.append(str(project_root))
from src.models.components.dpu_wrapper import (  # noqa: E402
    ResidualScaleHyperpriorDPUWrapper,
)
from src.models.components.res_scale_hyperprior_dpu import (  # noqa: E402
    ResidualScaleHyperpriorPatched,
)
from src.models.components.sar_simple_autoencoder import ResidualSimpleAE  # noqa: E402
from src.utils.constants import amp_max, amp_min  # noqa: E402
from src.utils.metrics import MerlinRDLoss  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run_dir",
    default=None,
    help="Path to the directory of the run. It must contains at the minimum the run config `.hydra/config.yaml` and the last checkpoint `checkpoints/last.ckpt`.",
)

# # Deprecated
# parser.add_argument(
#     "--data_dir",
#     default="/path/to/imagenet/",
#     help="Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation",
# )
# parser.add_argument(
#     "--model_dir",
#     default="/path/to/trained_model/",
#     help="Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth",
# )

parser.add_argument("--config_file", default=None, help="quantization configuration file")
parser.add_argument(
    "--subset_len",
    default=None,
    type=int,
    help="subset_len to evaluate model, using the whole validation dataset if it is not set",
)
parser.add_argument(
    "--batch_size", default=1, type=int, help="input data batch size to evaluate model"
)
parser.add_argument(
    "--quant_mode",
    default="calib",
    choices=["float", "calib", "test"],
    help="Quantization mode: float: evaluate float model, calib: quantize, test: evaluate quantized model.",
)
parser.add_argument(
    "--fast_finetune",
    dest="fast_finetune",
    action="store_true",
    help="fast finetune model before calibration",
)
parser.add_argument(
    "--deploy", dest="deploy", action="store_true", help="export xmodel for deployment"
)
parser.add_argument("--inspect", dest="inspect", action="store_true", help="inspect model")
parser.add_argument("--target", dest="target", nargs="?", const="", help="specify target device")

args, _ = parser.parse_known_args()


class CustomDataset(Dataset):
    def __init__(
        self,
        hdf5_path: Path,
    ):
        """Custom Dataset for loading patches from HDF5 file.

        Avoids problematic imports in TSXSSCDataset.
        """
        super().__init__()
        self.hdf5_path = hdf5_path

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as f:
            self.num_patches = f["patches"].shape[0]
            self.attrs = dict(f.attrs)

    def __len__(self):
        """Return the number of patches in the dataset."""
        return self.num_patches

    def __getitem__(self, idx):
        """Get a patch by index."""
        with h5py.File(self.hdf5_path, "r") as f:
            patch = torch.from_numpy(f["patches"][idx]).float()
            # The valid.h5 has 4 channels [real, imag, ADAM-NOC and MERLIN]. We discard the 2 references.
            patch = patch[:, :, 0:2]

            patch = torch.square(patch)
            patch = torch.log(patch + 1e-2)
            patch = (patch - 2 * amp_min) / (2 * amp_max - 2 * amp_min)

            return patch.permute(2, 0, 1)


def load_data(
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Load validation data loader."""
    dataset = CustomDataset(args.data_dir / "val.h5")
    if args.subset_len:  # random sampling method
        assert args.subset_len <= len(dataset)
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), args.subset_len)
        )
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return data_loader


def evaluate(
    quant_model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    float_model: torch.nn.Module = None,
    is_split_graph: bool = False,
) -> float:
    """Evaluate the model on validation dataset.

    Args:
        quant_model: The model to evaluate (quantized or wrapper).
        val_loader: DataLoader.
        loss_fn: Loss function.
        float_model: A float copy of the model used to generate intermediate inputs
                     if is_split_graph is True.
        is_split_graph: If True, inputs are manually split and fed as 4 separate tensors.
    """
    quant_model = quant_model.to(device)
    if float_model:
        float_model = float_model.to(device)
        float_model.eval()

    nb_images = 0
    loss_total = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs = data.to(device).float()

            if is_split_graph:
                # --- Split Graph Evaluation Strategy ---
                # We need to feed 4 inputs to the quant_model: (x, abs_y, z_hat, y_hat).
                # We generate the valid intermediate maps using the float_model.

                if float_model is None:
                    raise ValueError("float_model must be provided for split graph evaluation.")

                # 1. Prepare inputs (matches logic in dpu_wrapper.py Mode 1)
                x_real = inputs[:, :1, :, :]
                x_imag = inputs[:, 1:, :, :]

                # 2. Run Float Model to get intermediates
                # We can't just run float_model(inputs) because we need the internals.
                # using the patched model components directly:
                y_real = float_model.g_a(x_real)
                y_imag = float_model.g_a(x_imag)
                y = torch.cat((y_real, y_imag), dim=1)

                # Intermediate 1: abs_y
                abs_y = torch.abs(y)

                # Intermediate 2: z_hat (and z)
                z = float_model.h_a(abs_y)
                # In trace/calib mode we usually bypass entropy, so z_hat ~= z
                z_hat = z

                # Intermediate 3: y_hat
                # For calibration of g_s, we need y_hat. In trace mode y_hat ~= y
                y_hat = y

                # 4. Run Quantized Model with 4 inputs
                # The output in Mode 2 is a tuple: (y_out, z_out, scales_out, x_out)
                # For g_s input, we take the real part (first half channels) as approximation,
                # effectively running g_s on "real" data.
                y_hat_effective = y_hat[:, : y_hat.shape[1] // 2, :, :]

                # For g_a input (first arg), we pass x_real (1 channel)
                _ = quant_model(x_real, abs_y, z_hat, y_hat_effective)

                # We cannot compute a meaningful loss in Split Mode because the graph is broken.
                current_loss = 0.0

            else:
                # --- Standard Mode 1 (Connected) ---
                output = quant_model(inputs)
                if i == 0:
                    # debug print
                    pass
                loss = loss_fn(output, inputs)
                current_loss = loss["distortion"].item()

            loss_total += current_loss
            nb_images += inputs.size(0)

    return loss_total / nb_images if nb_images > 0 else 0.0


def try_match_run_dir_with_existing_runs() -> Path:
    """Try to match run_dir with existing runs."""
    dir_prefix = Path("DDC_FPGA/logs/train/sar_ddc/")
    existing_runs = {
        "ResSHyp-relu_42_merlinʎ100_lr5e-05_b12": "hyperprior/multiruns/2025-11-18_09-45-37/6",
        "ResAE-relu_42_merlinʎ100_lr5e-05_b12": "simple_ae/multiruns/2025-11-16_19-34-55/3/",
        "ResAE-relu_42_merlinʎ100_lr5e-05_b12_no-out-pad": "simple_ae/runs/2025-11-19_15-22-34",
        "ResSHyp_export_dpu_test": "hyperprior_dpu/runs/2025-12-01_13-36-00",
    }
    run_path = existing_runs.get(args.run_dir, None)
    if run_path is None:
        raise RuntimeError(f"Cannot find matching run for {args.run_dir}!")
    return dir_prefix / run_path


def check_and_enforce_arguments():
    """Check and enforce arguments."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        run_dir = try_match_run_dir_with_existing_runs()

    args.model_dir = run_dir / "checkpoints"
    config_path = run_dir / ".hydra" / "config.yaml"
    # read hydra config to find data_dir
    args.hydra_conf = OmegaConf.load(config_path)
    assert type(args.hydra_conf) is DictConfig

    args.data_dir = Path(args.hydra_conf.data.get("hdf5_dir", None))
    if not args.data_dir or not args.data_dir.exists():
        raise RuntimeError(f"args.data_dir {args.data_dir} does not exist!")

    if args.quant_mode != "test" and args.deploy:
        warnings.warn(
            "Exporting xmodel needs to be done with `--quant_mode test`. Setting deploy to False."
        )
        args.deploy = False
    if args.inspect and args.quant_mode == "float" and args.batch_size != 1:
        warnings.warn("Inspecting model needs batch size to be 1. Enforcing it.")
        args.batch_size = 1
    if args.deploy and (args.batch_size != 1 or args.subset_len != 1):
        warnings.warn(
            "Exporting xmodel needs batch size to be 1 and only 1 iteration of inference. Enforcing them."
        )
        args.batch_size = 1
        args.subset_len = 1


def debug_forward_execution(model, inputs, desc):
    """Helper function to inspect which modules are executed during a forward pass.

    This helps diagnosing mismatch errors between Calibration and Deployment graphs.
    """
    print(f"\n[DEBUG] --- Forward Execution Trace: {desc} ---")
    print(f"[DEBUG] Input shapes: {[i.shape for i in inputs]}")

    hooks = []

    def get_hook(name):
        """Create a forward hook that prints module execution."""

        def hook(module, input, output):
            """Forward hook function."""
            # Print only leaf modules or interesting ones to avoid spam
            if len(list(module.children())) == 0 or "Sequential" in module.__class__.__name__:
                print(f"[DEBUG] Executing: {name} (Type: {module.__class__.__name__})")

        return hook

    # Register hooks on all named modules
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(get_hook(name)))

    # Run forward pass (no grad)
    try:
        with torch.no_grad():
            model(*inputs)
    except Exception as e:
        print(f"[DEBUG] Forward pass crashed: {e}")
    finally:
        # Cleanup
        for h in hooks:
            h.remove()
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    check_and_enforce_arguments()

    # ---- Find the model -----
    model_path = args.model_dir / "last.ckpt"
    # We can use hydra.utils.instantiate() because it would try to import lightning, which we don't have int this Vitis-Ai container.
    # So we parse the network._target_ and instantiate the model directly.
    model_name = args.hydra_conf.model.net._target_.split(".")[-1]
    model_params = args.hydra_conf.model.net
    model_params.pop("_target_")

    full_model = None  # Hold reference to float model for split graph execution wrapping

    # For DPU export / inspection, always use the patched model with export_dpu=True.
    # We still load weights from the training-time ResidualScaleHyperprior checkpoint.
    if model_name in ("ResidualScaleHyperpriorPatched"):
        model_params["export_dpu"] = True
        print(f"Using ResidualScaleHyperpriorPatched for DPU with params: {model_params}.")
        full_model = ResidualScaleHyperpriorPatched(**model_params).to(device)

        # Load weights into full model
        checkpoint = torch.load(model_path)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items()}
            full_model.load_state_dict(state_dict, strict=False)
        else:
            full_model.load_state_dict(checkpoint, strict=False)

        # Disable gradients for all parameters to avoid VAI_Q trace errors
        for param in full_model.parameters():
            param.requires_grad = False

        print("Transforming to ResidualScaleHyperpriorDPUWrapper (Calibration Mode)...")
        model = ResidualScaleHyperpriorDPUWrapper(full_model)

    elif model_name == "ResidualSimpleAE":
        # Simple AE has no entropy modules; nothing special needed
        print(f"Using ResidualSimpleAE with params: {model_params}.")
        model = ResidualSimpleAE(**model_params).to(device)

        checkpoint = torch.load(model_path)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Model {model_name} not recognized for DPU compilation!")

    print("Loaded a checkpoint successfully.")

    model.eval()

    # Define dummy inputs for both Single and Split modes
    # Shapes must match the architecture (N=128, M=256)

    is_wrapper = isinstance(model, ResidualScaleHyperpriorDPUWrapper)

    # Split Mode Inputs
    if is_wrapper:
        # x: [B, 1, H, W] (g_a expects 1 channel)
        x_dumb = torch.randn(args.batch_size, 1, 256, 256).to(device)
        # abs_y: [B, M, H/16, W/16]
        abs_y_dumb = torch.randn(args.batch_size, 256, 16, 16).to(device)
        # z_hat: [B, M, H/128, W/128] (16/8 = 2)
        z_hat_dumb = torch.randn(args.batch_size, 256, 2, 2).to(device)
        # y_hat: [B, N, H/16, W/16]
        y_hat_dumb = torch.randn(args.batch_size, 128, 16, 16).to(device)

        # Use Split Graph for Vitis-AI Quantization (Consistent for Calib & Test)
        dummy_inputs = (x_dumb, abs_y_dumb, z_hat_dumb, y_hat_dumb)
        print("Using Split Graph (4 inputs) for Vitis-AI Quantization.")
    else:
        # Standard Mode (2 channels for Calib Mode 1 of wrapper OR normal model)
        # Wait, if it is NOT a wrapper, it might be SimpleAE which takes 2 channels?
        # Or if it is wrapper in mode 1, it takes 2 channels.
        # But here we decided: IF wrapper -> Use Split Mode.
        # So "else" is for SimpleAE. SimpleAE takes 2 channels?
        # Let's check SimpleAE definition. It usually takes 2 channels.
        x_dumb = torch.randn(args.batch_size, 2, 256, 256).to(device)

    # ----- inspect -----
    if args.quant_mode == "float":
        quant_model = model
        if args.inspect:

            def inspect_activations(module, prefix=""):
                for name, child in module.named_children():
                    fullname = f"{prefix}.{name}" if prefix else name
                    cls_name = child.__class__.__name__
                    if "GDN" in cls_name or "ReLU" in cls_name:
                        print(f"[{cls_name}] found at {fullname}")
                    inspect_activations(child, fullname)

            if hasattr(model, "g_a"):
                inspect_activations(model.g_a, "g_a")
                inspect_activations(model.g_s, "g_s")

            if not args.target:
                raise ValueError("Target must be specified for Inspector.")
            from pytorch_nndct.apis import Inspector  # type: ignore

            inspector = Inspector(args.target)
            inspector.inspect(model, dummy_inputs, device=device)
            sys.exit()
    else:
        # ----- Setup -----
        if args.quant_mode == "calib":
            # [DEBUG] Trace graph before calibration
            debug_forward_execution(
                model, dummy_inputs, f"Calibration Mode ({len(dummy_inputs)} Inputs)"
            )

        quantizer = torch_quantizer(
            args.quant_mode,
            model,
            dummy_inputs,
            device=device,
            quant_config_file=args.config_file,
            target=args.target,
        )
        quant_model = quantizer.quant_model

    # Get loss after evaluation @TODO: Write MerlinRDLoss functional
    loss_params = args.hydra_conf.model.criterion
    loss_params.pop("_target_")
    loss_fn = MerlinRDLoss(**loss_params).to(device)

    # ----- Load data -----
    val_loader = load_data()
    # print the shape of the data
    for data in val_loader:
        print(f"Data batch shape: {data.shape}")
        break

    # fast finetune model or load finetuned parameter before test
    if args.fast_finetune:
        ft_loader = load_data()
        extra_kwargs = {"float_model": full_model, "is_split_graph": True} if is_wrapper else {}
        if args.quant_mode == "calib":
            quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn), **extra_kwargs)
        elif args.quant_mode == "test":
            quantizer.load_ft_param()

    # @TODO: keep here scores of float model to print and compare
    # Float model: Loss after evaluation: 5545.23803125
    # Quantized model: Loss after evaluation: 5776.80296875
    print(f"Evaluating model in '{args.quant_mode}' mode...")
    extra_eval_kwargs = {"float_model": full_model, "is_split_graph": True} if is_wrapper else {}
    loss_gen = evaluate(quant_model, val_loader, loss_fn, **extra_eval_kwargs)
    print(f"Loss after evaluation: {loss_gen}")

    # handle quantization result
    if args.quant_mode == "calib":
        quantizer.export_quant_config()
    if args.deploy:
        if is_wrapper:
            print("Exporting split xmodel for Hybrid Inference...")
            quantizer.export_xmodel(deploy_check=False)
        else:
            # Standard deployment for SimpleAE or others
            # @TODO: I should set output_dir to script location + quantize_results/
            quantizer.export_torch_script()
            # quantizer.export_onnx_model()
            quantizer.export_xmodel(deploy_check=True)

        # @TODO: Print the size of the xmodel's input buffer

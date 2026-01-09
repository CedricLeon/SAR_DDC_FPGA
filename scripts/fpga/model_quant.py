"""
This script will be used to quantize a PyTorch model for FPGA deployment.
It is based on Vitis-AI resnet18 PyTorch model quantization example, available at: https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html#pytorch-tutorial

@TODO:
- I might have to have this script in Vitis-AI folder, because I need to start the docker there
- Guess the data_dir from model configuration file (to avoid messing up)
- Write functional MerlinRDLoss function to compute the loss after evaluation
- Write how my evaluation will work

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
# from src.data.components.sar_dataset import TSXSSCDataset  # noqa: E402
from src.models.components.res_scale_hyperprior_dpu import (  # noqa: E402
    ResidualScaleHyperpriorPatched,
)
from src.models.components.sar_simple_autoencoder import ResidualSimpleAE  # noqa: E402
from src.utils.constants import amp_max, amp_min  # noqa: E402
from src.utils.metrics import MerlinRDLoss  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module
) -> float:
    """Evaluate the model on validation dataset."""
    # @TODO: figure out what I want to evaluate
    model = model.to(device)
    nb_images = 0
    loss_total = 0
    for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        input = data.to(device).float()
        output = model(input)
        if i == 0:
            print(f"{input.shape=}, {output['x_hat'].shape=}")
        loss = loss_fn(output, input)
        loss_total += loss["distortion"].item()
        nb_images += input.size(0)
    return loss_total / nb_images


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


if __name__ == "__main__":
    check_and_enforce_arguments()

    # ---- Find the model -----
    model_path = args.model_dir / "last.ckpt"
    # We can use hydra.utils.instantiate() because it would try to import lightning, which we don't have int this Vitis-Ai container.
    # So we parse the network._target_ and instantiate the model directly.
    model_name = args.hydra_conf.model.net._target_.split(".")[-1]
    model_params = args.hydra_conf.model.net
    model_params.pop("_target_")

    # For DPU export / inspection, always use the patched model with export_dpu=True.
    # We still load weights from the training-time ResidualScaleHyperprior checkpoint.
    if model_name in ("ResidualScaleHyperpriorPatched"):
        model_params["export_dpu"] = True
        print(f"Using ResidualScaleHyperpriorPatched for DPU with params: {model_params}.")
        model = ResidualScaleHyperpriorPatched(**model_params).cpu()
    elif model_name == "ResidualSimpleAE":
        # Simple AE has no entropy modules; nothing special needed
        print(f"Using ResidualSimpleAE with params: {model_params}.")
        model = ResidualSimpleAE(**model_params).cpu()
    else:
        raise ValueError(f"Model {model_name} not recognized for DPU compilation!")

    checkpoint = torch.load(model_path)
    ckpt_type = "regular PyTorch"
    if "state_dict" in checkpoint:
        # Lightning prefixes parameters with "model." or similar, need to remove that
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items()}
        ckpt_type = "Lightning"
    message = model.load_state_dict(checkpoint, strict=False)
    print(f"     Loaded a {ckpt_type} checkpoint: {message}")

    model.eval()

    # ----- inspect -----
    input_data = torch.randn([args.batch_size, 2, 256, 256])
    if args.quant_mode == "float":
        quant_model = model
        if args.inspect:
            if not args.target:
                raise ValueError("Target must be specified for Inspector.")
            from pytorch_nndct.apis import Inspector  # type: ignore

            # torch.onnx.export(
            #     model,
            #     (input_data,),
            #     f"original_{model_name}_skeleton.onnx",
            #     export_params=False,
            #     opset_version=17,
            #     do_constant_folding=True,
            #     input_names=["input"],
            #     output_names=["output"],
            # )

            inspector = Inspector(args.target)
            inspector.inspect(model, (input_data,), device=device)

            sys.exit()
    else:
        # ----- Setup -----
        quantizer = torch_quantizer(
            args.quant_mode,
            model,
            (input_data,),
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
        if args.quant_mode == "calib":
            quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
        elif args.quant_mode == "test":
            quantizer.load_ft_param()

    # @TODO: keep here scores of float model to print and compare
    # Float model: Loss after evaluation: 5545.23803125
    # Quantized model: Loss after evaluation: 5776.80296875
    loss_gen = evaluate(quant_model, val_loader, loss_fn)
    print(f"Loss after evaluation: {loss_gen}")

    # handle quantization result
    if args.quant_mode == "calib":
        quantizer.export_quant_config()
    if args.deploy:
        # @TODO: I should set output_dir to script location + quantize_results/
        quantizer.export_torch_script()
        # quantizer.export_onnx_model() # Get an ERROR: Exporting the operator 'aten::erfc' to ONNX opset version 17 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues
        quantizer.export_xmodel(deploy_check=True)

        # @TODO: Print the size of the xmodel's input buffer

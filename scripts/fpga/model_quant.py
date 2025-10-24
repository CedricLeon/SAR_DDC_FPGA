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
import random
import sys
from pathlib import Path

import torch
from pytorch_nndct.apis import torch_quantizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.components.sar_dataset import TSXSSCDataset
from src.models.components.sar_hyperprior import ResidualScaleHyperprior
from src.utils.metrics import MerlinRDLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    default="/path/to/imagenet/",
    help="Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation",
)
parser.add_argument(
    "--model_dir",
    default="/path/to/trained_model/",
    help="Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth",
)
parser.add_argument("--config_file", default=None, help="quantization configuration file")
parser.add_argument(
    "--subset_len",
    default=None,
    type=int,
    help="subset_len to evaluate model, using the whole validation dataset if it is not set",
)
parser.add_argument(
    "--batch_size", default=32, type=int, help="input data batch size to evaluate model"
)
parser.add_argument(
    "--quant_mode",
    default="calib",
    choices=["float", "calib", "test"],
    help="quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model",
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


def load_data(
    data_dir: Path,
    **kwargs,
):
    """Load validation data loader."""
    dataset = TSXSSCDataset(data_dir / "val.h5", transform=None)
    if args.subset_len:  # random sampling method
        assert args.subset_len <= len(dataset)
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), args.subset_len)
        )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return data_loader


def evaluate(model, val_loader, loss_fn):
    """Evaluate the model on validation dataset."""
    # @TODO: figure out what I want to evaluate
    model.eval()
    model = model.to(device)
    nb_images = 0
    loss_total = 0
    for i, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss_total += loss.item()
        nb_images += images.size(0)
    return loss_total / nb_images


if __name__ == "__main__":
    # ----- Parse/Preprocess arguments -----
    data_dir = Path(args.data_dir)
    if args.quant_mode != "test" and args.deploy:
        args.deploy = False
        print(
            r"Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!"
        )
    if args.deploy and (args.batch_size != 1 or args.subset_len != 1):
        print(
            r"Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!"
        )
        args.batch_size = 1
        args.subset_len = 1
    # ---- Find the model -----
    # ~/dev/DDC_FPGA/logs/train/sar_ddc/hyperprior/runs/2025-06-10_13-47-27
    model_path = Path(args.model_dir) / "last.ckpt"
    model = ResidualScaleHyperprior().cpu()  # model is a nn.Module with a forward() method
    model.load_state_dict(torch.load(model_path))

    # ----- inspect -----
    input = torch.randn([args.batch_size, 1, 256, 256])
    if args.quant_mode == "float":
        quant_model = model
        if args.inspect:
            if not args.starget:
                raise ValueError("Target must be specified for Inspector.")
            from pytorch_nndct.apis import Inspector

            inspector = Inspector(args.target)
            inspector.inspect(model, (input,), device=device)

            sys.exit()
    else:
        # ----- Setup -----
        quantizer = torch_quantizer(
            args.quant_mode,
            model,
            (input),
            device=device,
            quant_config_file=args.config_file,
            target=args.target,
        )
        quant_model = quantizer.quant_model

    # Get loss after evaluation @TODO: Write MerlinRDLoss functional
    loss_fn = MerlinRDLoss(metric="mse", lmbda=0.01).to(device)

    # ----- Load data -----
    val_loader = load_data(data_dir)

    # fast finetune model or load finetuned parameter before test
    if args.finetune:
        ft_loader, _ = load_data(data_dir)
        if args.quant_mode == "calib":
            quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
        elif args.quant_mode == "test":
            quantizer.load_ft_param()

    # @TODO: keep here scores of float model to print and compare
    loss_gen = evaluate(quant_model, val_loader, loss_fn)
    print(f"Loss after evaluation: {loss_gen}")

    # handle quantization result
    if args.quant_mode == "calib":
        quantizer.export_quant_config()
    if args.deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel(deploy_check=False)

# Synthetic Aperture Radar (SAR) Despeckling and Data Compression (DDC) on Field Programmable Gate Arrays (FPGA)
<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

Yes, that's a lot of acronyms. But now you know why it's called DDC_FPGA.
This project implements the solution presented by Amao-Oliva et al. [1] available at [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0924271624004866) on FPGA.

### TODOs
*I'll use this section as a TODO list, including ideas for future projects.*
- [ ] "NWML" warning, see [NVML is the NVIDIA Management Library and is used on NVIDIA GPUs](https://discuss.pytorch.org/t/cant-initialize-nvml-error-with-rvc-project/194206)
- [ ] Spatial_split dataset
- [ ] Incorporate validation_big_patch generation in [TSX_dataset_creation.py](scripts/TSX_dataset_creation.py)
- [ ] Settle on metric accumulation strategy: "mean" or "sum", and derive $\lambda$ range accordingly

### Long-term Experiments/Upgrades
- [ ] Maybe there is a way to avoid the concatenation and average latent representations before hyperprior ???
- [ ] Experiment with data preprocessing. Asymmetric percentiles, e.g., p0.1% and p95% (For log-intensity, ~0.0 and ~11.6, something like MERLIN's m and M).

### Method
The pipeline relies on Pytorch Ligthning on [Compressai](https://github.com/InterDigitalInc/CompressAI) [2] to implement Hyper-autoencoders solutions based on Johannes Ballé's work [3-5].
In addition, the despeckling task is inspired from MERLIN's self-supervised training pipeline [6].

#### MERLIN Theory
**The big picture (mostly written by ChatGPT)**:
Dalsasso et al. introduce MERLIN, a fully self-supervised strategy for training deep despeckling networks directly on single-look complex (SLC) SAR images. By exploiting Goodman’s speckle model—which shows that the real and imaginary components of an SLC pixel are two independent, Gaussian-distributed realizations with variance proportional to the local reflectivity $r$—they train a U-Net to predict pixel-wise variance maps (i.e., the effective “blurred” reflectivity $r$) from one component (say, the real part) and evaluate the loss on the other component (the imaginary part).


Taking a SLC SAR image with Real part $\tilde{a}$ and Imaginary part $\tilde{b}$ (We use tilde notation to indicate the transformation from the SAR transfer function **H**, see MERLIN Eq.(4) and (5)). Its intensity $I$ is
$$
I = \tilde{a}^2 + \tilde{b}^2
\tag{1}
$$
MERLIN aims at depesckling, i.e., reconstructing the underlying reflectivity image $\tilde r$, with a self-supervised model $f_\theta()$ as $\frac{f_\theta(\tilde{a}^2) + f_\theta(\tilde{b}^2)}{2}$.
To achieve this, the model is trained with the loss function below, where $k$ represent the iteration over pixels:
$$
\mathcal{L}(\tilde{r}, \tilde{b}) = \sum_k \frac{log(\tilde{r}_k)}{2} + \frac{\tilde{b}_k^2}{\tilde{r}_k}
\tag{2a}
$$
In practice, it is useful to work in log-scale to reduce the dynamic range. We use the check notation to represent log-scaled variables: $\check r = log~\tilde r$, $\check a = log|\tilde a|$ and $\check b = log|\tilde b|$.
> Note: The absolute value operator |.| is only used for mathematical correctness as $\tilde a$ and $\tilde b$ can have negative values. In practice it is not needed as these values are squared before being fed to the network. As a last detail, it is also necessary to add a small $\epsilon$ to $\tilde a$ or $\tilde b$, to avoid $log(0)$. In python, this is done with `1e-6` or, preferably, `np.spacing(1)`.

Working in log-scale implies to modify the loss:
$$
\mathcal{L}(\check r, \check b) = \sum_k \frac{\check r_k}{2} + exp(2 \check b_k - \check r_k)
\tag{2b}
$$

**A few mode details**:
- **Further normalization**. In addition of the log-scale, it is beneficial to "normalize the images using a fixed affine transform".
> What this mean is using the minmax formula, but not with the minimum and maximum values of the image (because of the strong outliers, the whole distribution would end up being very narrow and the network would struggle differentiating values). Instead percentile values are used, typically 5 and 95%.

- **Misconception about the data range of the output**. As (most of) the data lies between $[0;1]$, one could expect the reconstruction of the network be in the same interval. However, the network learns to map from noisy realizations of $\tilde a \sim \mathcal{N}(0,r/2)$ to the total reflectivity $r$, not to $\frac{r}{2}$. Same for $\tilde b$.
> This means that to compare 2 images of the same scale one must visualize the Intensity $I = \tilde{a}^2 + \tilde{b}^2$ and a single prediction , e.g., $f_\theta(\tilde{a}^2)$. During inference both network estimations are averaged to decrease the variance of the reconstructed reflectivity, but if a simple proxy is needed, one could use only one of the reconstructions.

- **An $ln(2)$ offset**. @TODO. One thing I still do not undestand comes from the loss optimization. If the network tries to optimize Eq. (2b), that I simplify with $\hat x$ the reconstruction and $y$ the target:
$$
\text{Optimizing}~f(\hat x, y) = \frac{\hat x}{2} + exp(2 y - \hat x) \\
\text{Means finding where}~f'(\hat x, y): \frac{1}{2} - exp(2 y - \hat x) = 0 \\
\Leftrightarrow 2y - \hat x = ln(\frac{1}{2}) \\
\Leftrightarrow \hat x = 2y + ln(2)
$$
> Given that we reconstruct the estimated reflectivity and not half of it $2y$ makes kind of sense, however, the ln(2) offset does not have an explanation to me. 

### Data
TerraSAR-x StripMap (SM) SSC (Single Look Slant Range Complex) images downloaded from [ESA's platform](https://earth.esa.int/eogateway/catalog/terrasar-x-esa-archive).
> Submitting a form is required to access the data (2 days max delay).

#### Pre-processing
`script/TSX_dataset_creation.py` creates `hdf5` datasets more convenient for training that re-processing and patchifying the entire TSX SSC images everytime.
In particular, each .cos file present in `data/TSX_cos_files/` is open, images are patchified, symmetrized, strong scatterers are preserved, normalized, and the whole set of resulting patches is split in training/validation/test datasets.
The data is pre-processed into train/val/test HDF5 files using the `script/TSX_dataset_creation.py`. This script has many options, use `--help` for details.

#### Datasets naming
I'll try to follow the dataset naming conventions below: `<split_type><nb_images>_<preservation>_<normalization>.hdf5`, where:
- `split_type` is how the patches where split ("randomsplit" or "spatialsplit")
- `nb_images` corresponds to the number of `.cos` files used for the dataset (typically 5)
- `preservation` indicates if strong point-like scatterers were preserved following (@TODO add equation in [[Math.md]]) and the threshold, for example "nopres" or "pres60dB".
- `normalization` consists of `norm<normalization_percentiles><log_base><clipped>`
  - `normalization_percentiles` indicates if min-max normalization was performed on the patches, and which percentiles were used as "min" and "max". For example, "nonorm" or "norm1".
  - `log_base` the logarithmic base was used in the normalization, either "db" (`np.log10()`) or "nat" (natural: `np.log()`)
  - `clipped` is "clip" or "" depending if the data was clipped to [0,1] or not. If `normalization_percentiles=0` clipping is deactivated by default, as after a minmax normalization the data lies already in [0,1]
Examples:
- "randomsplit5_pres60dB_norm5db" was processed with preservation of scatterers with signals above 60dB, the patches were placed in log10 base before being "min-maxed" with p5 and p95 (i.e., value 0 corresponds to p5 and value 1 to p95)
- "randomsplit5_nopres_norm0nat" was processed without scatterer preservation but with normalization to natural logarithm and traditional min-max (0 means 0%)

#### Training
After downloading, unzipping, accessing the `.cos` files (deep in the archive in `IMAGEDATA/`), and copying all CoSAR file format in a common folder, e.g., `data/TSX_cos_files/`, datasets are created to ease training.
> The `data` folder is ignored by Git, but typically contains softlinks to the datasets (to avoid several copies of big files).

==To place somewhere else==:
`dataset_creation.py` creates `hdf5` datasets more convenient for training that re-processing and patchifying the entire SAR SLC images everytime.
In particular, each .cos file present in `data/TSX_cos_files/` is open, images are symmetrizeda and patchified, and the whole set of resulting patches is split in training/validation/test datasets.
> Following MERLIN's pipeline: Patches are not normalized between 0 and 1. The log-normalization happens before feeding them to the network. They are also de-normalized afterwards.

#### Testing / Inference
@TODO, implement a script that automatically find all TSX archives, unzip them, extract the `.cos` file, pre-process the data,loads the model, despeckle and compress the images, and why not reconstruct them. (Will most likely follow MERLIN's inference workflow)

### Hardware implementation
Vitis AI [7]

#### References
[1] Joel Amao-Oliva, Nils Foix-Colonier, Francescopaolo Sica. (2024). Joint compression and despeckling by SAR representation learning. ISPRS Journal of Photogrammetry and Remote Sensing.
[2] J. Bégaint, F. Racapé, S. Feltman, and A. Pushparaja, “CompressAI: a PyTorch library and evaluation platform for end-to-end compression research,” Nov. 05, 2020, arXiv: arXiv:2011.03029. doi: 10.48550/arXiv.2011.03029.
[3] D. Minnen, J. Ballé, and G. D. Toderici, “Joint Autoregressive and Hierarchical Priors for Learned Image Compression,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2018. Accessed: Jun. 14, 2023.
[4] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” presented at the International Conference on Learning Representations, Feb. 2018. Accessed: Apr. 10, 2024.
[5] J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-end Optimized Image Compression,” Mar. 03, 2017, arXiv: arXiv:1611.01704. doi: 10.48550/arXiv.1611.01704.
[6] Dalsasso, E., Denis, L., & Tupin, F. (2022). As if by magic: Self-supervised training of deep despeckling networks with MERLIN. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–13. https://doi.org/10.1109/TGRS.2021.3128621
[7] AMD Vitis™ AI Software. (2019). AMD. https://www.amd.com/en/products/software/vitis-ai.html



# Original Template README
Below is the original README for the [template](https://github.com/CedricLeon/Setup_Lightning_Hydra_template). I keep it for the moment.

## Description

This repo is just my fork from [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) where I modified some setup parameters to be ready to go directly after cloning.
Reference the **Modified from template** section to see the changes. otherwise the main modifications are the following:

- I set up `hydra-submitit-launcher` for an easier usage of SLURM, and add example config setups for clusters (JUWELS, Terrabyte to come)
- I commit to use Weight & Biases (W&B) logger. Other loggers are still possible to use, but everything is setup by default for W&B.
- I use [wandb_osh](https://github.com/klieret/wandb-offline-sync-hook) to support offline, real-time logging of my runs on W&B. In this template, setting up `wandb_osh` is as easy as that:
  - Switch `logger.wandb.offline` to `True`
  - Have the "Farm" running on the login node, i.e., with the command `wandb-osh`

## Installation

### Setting up Git

There are 2 different ways you can setup your new repository: by keeping track of the template, or by starting a fresh new git repo with all the files from the template.

> If you plan to host your code on DLR GitLab, you should make sure that when you create the new repository you create it as a "blank project", select **<your_user>** and not **<your_group>** in the Project URL and uncheck "Initialize repository with a README".
> Also, you should use the HTTPS URLs.

In both cases you first need to clone the template and rename the folder with `<your_project_name>`:

```bash
# Clone the template
git clone https://github.com/CedricLeon/Setup_Lightning_Hydra_template.git
# Rename the folder with your project name
mv Setup_Lightning_Hydra_template/ <your_project_name>/
cd <your_project_name>/
```

#### Re-initializating Git history

Then you can either delete the remote and commit history of the template, this is the most straightforward way:

```bash
# Reset the git repository
rm -rf .git/
git init --initial-branch=main
# Add your remote
git remote add origin <your_remote_URL>
# Stage and commit all files + set origin main as upstream
git add .
git commit -m "Initial commit"
git push --set-upstream origin main
```

#### Keeping track of the template Git history (*homemade*)

Or you can keep the remote but rename it to `template` and add a new `origin`.

I describe how to do that below, but you should know that it is just a homemade version of template repository from GitHub. It is less clean, but allows to host the new repo on a server that isn't GitHub (I didn't find a way to do that using the GitHub template feature). *If someone has a cleaner way of doing it, please enlight me.*

```bash
# Rename the template remote
git remote rename origin template
# Add your new repository remote. So, yes, you need to create it before
git remote add origin <your_remote_URL>
git remote -v

# Synchronize your (empty new repo) with a rebase to avoid non-fast-forward errors
git pull --rebase origin main
# Push the commit history and all the template files on the new repo (also set the origin/main branch as upstream)
git push --set-upstream origin main
```

### Set your conda environment

```bash
# Force python version 3.11 for compatibility reasons (pytorch)
conda create -n <your_env_name> python=3.11
conda activate <your_env_name>

# /!\ install pytorch with GPU support, see https://pytorch.org/get-started/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt
```

### Optional: Setup `pre-commit` hooks

If you don't know [pre-commit](https://pre-commit.com/) hooks, they do exactly what the name suggests, avoiding you to commit stupid typos or performing code linting for you in the background. Check the docs for more details.

So, in case you deleted `.git` after cloning the template, you have to reinstall pre-commit.
It's also a good idea to run it against all files (if you have any) for the first time.

```bash
pre-commit install
pre-commit run --all-files
```

You can test that pre-commit is nicely setup with a dummy commit, or just by committing the changes of the next sections.

> Note: if you are using VSCode commit system, the output logs are redirected towards the `OUTPUT/Git` console. Nevertheless, you should still get an error message if you messed something. Spoiler: the error message is not helpful, but redirects you towards the git logs.

### Personalize project template parameters

I have fixed some parameters that are project specific with generic names (e.g., `logger.wandb.project: "lightning-hydra-template"`). Here is a list you should check and replace:

- **Documentation**: Change the title of this `README.md` (and most likely also delete the crap I wrote 😉)
- **W&B**: In `configs/logger/wandb.yaml`, `team: "my-wandb-team"` and `project: "lightning-hydra-template"`
- **Submitit** (if you plan to use multiruns):
  - In `configs/hydra/launcher/` change your account settings in the different cluster setups: `account: "your_juwels_project"` (if necessary, also update your favorite `partition`).
  - You can specify the launcher through your command, with the option `hydra.launcher.partition=juwels_single_gpu` for example
  - Otherwise, in the experiment file, add a configuration for `hydra-submitit-launcher`:

```yaml
# Just after defaults:
  - override /hydra/launcher: juwels_single_gpu # for example
```

As a general comment, I advise to run a mock run (/!\ not with `debug=fdr` /!\\, it hides most of the config) and have a careful look at your config. @TODO

### Optional: Test the environment

You can try running a 10 epochs training of a SimpleDenseNet on MNIST classification problem to check if everything runs smooth. If you already logged on W&B on your system you should not need to do anything else for the setup to be complete.

```bash
# Run on cpu by default
python src/train.py experiment=example
# If on a cluster, you can open an interactive session and run on gpu
python src/train.py experiment=example trainer=gpu
# Otherwise, you can run in "multirun" mode from the logging node
# /!\ Remember to specify the submitit launcher, and if necessary to set the run `offline`, otherwise W&B will crash the run /!\
python src/train.py -m experiment=example trainer=gpu hydra.launcher.partition=develbooster # or logger.wandb.offline=True
```

## Usage / Run

@TODO: refine the usage examples with how I use experiments, etc.

### Classic usage

The method I describe below is **my** preferred way of using this template. Of course, that's only a theory and you are free to organize yourself differently, the repository is very flexible.
However, after trying out different setups I often found myself lost, e.g., trying to find out why a parameter kept its old value when I was overriding it. In any case, [Hydra documentation](https://hydra.cc/docs/patterns/configuring_experiments/) is your best friend.
Now that you are warned, here is are my best practices.

In short, I recommend always creating runs from an experiment config. This enforces better hierarchy and organization, while having the advantage of grouping "all" modifications in a single file, making modifications easy.

See below an example to run with a chosen experiment configuration from [configs/experiment/](configs/experiment/):

```bash
python src/train.py experiment=example
```

### Overriding HYDRA config from CLI

From here, you can override minor parameters from the CLI for a quick check or a specific run:

```bash
python src/train.py experiment=example trainer.max_epochs=2
```

Whenever you find yourself, running several times similar commands with a high number of overrides, this is probably a good time to create a new `experiment.yaml`.

### Overriding a full config group

Sometimes, you might want to change big parts of your experiment config without wanting to redefine a new experiment, then, you can override Config Group options.
Examples non-exhaustively include estimating results on a different dataset, checking run time on a different hardware, or logging to csv because you're a boomer.

```bash
# Train on CPU
python src/train.py experiment=example trainer=cpu
# Quickly test another dataset
python src/train.py experiment=example data=kodak
# Change the logger
python src/train.py experiment=example logger=csv
```

### Debugging

Debugging is a instance of the previous case, where you override the debug package from the CLI. However, it's so common and important it deserves its own section.
Firstly, whenever you specify `debug` **there won't be any logging or callbacks** and the run will be executed without multithreading on CPU.
The best example is the `fast_dev_run` option of the Lightning Trainer which will run 1 step of training, validation and test. This is what I use 99% of the time.

```bash
python src/train.py experiment=example debug=fdr
```

If you still want some logging, or want to debug on GPU, etc. you can always specify that **after** your debug setup.

```bash
python src/train.py experiment=example debug=default trainer=gpu
```

## Modified from template

This section simply lists the major changes I brought to the original template [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). It's also here that I give a big shoutout to **ashleve**, in addition of the impressive work behind such repo, he is also on most of the Issues and PR I came across when I was setting this fork.

### New features

- Add deterministic training support (can be unset from config)
- Add the W&B offline management using `wandb_osh` (automatically adds the Lightning Callback when the run is set offline)
- Redirect logs to subdirectories specific for each experiment (see `task_name`)
- Automate job submission on cluster using `hydra-submitit-launcher` through `--multirun` mode

### More dependencies

- Uncomment my favorite logger in `environment.yaml` (**wandb**) as well as in `requirements.txt`
- Add additional requirements:
  - `hydra-submitit-launcher`
  - `torchgeo`
  - `wandb_osh` (Wandb Offline Sync Hook)
- Uncomment `sh` in `requirements.txt` to allow the tests in `test_sweeps.py`

### CI/CD and Testing

- *To execute all tests (require GPUs)*: execute `pytest` on a compute node (e.g., with an interactive session) to validate `@RunIf(min_gpus=1)` in `test_train.py` (make sure Pytorch is installed with GPU support)
**=> Get all tests to be executed and None skipped.**
- I removed MacOS and Windows deployment test, as well as most of the different versions of python tested (reason: save compute resources)
- The tests to be executed in CI/CD are the `"not slow"` ones, for the same reason mentioned above

## Features to come, @TODO

- [ ] Add a submitit setup for Terrabyte
- [ ] Increase test coverage, and provide classic examples to test Lightning Datamodule and Modules.
- [ ] Upgrade and "automate" the `task_name` parameter generation:
  - Either by using a specific name parameter in each Config Group option (config file) and `**kwargs` in the corresponding Modules.
  - Or by making it general and global in the "root" config file using Hydra interpolation system. Not that easy because it's impossible to interpolate in the Default List, see this [stackoverflow](https://stackoverflow.com/questions/67280041/interpolation-in-hydras-defaults-list-cause-and-error).
- [ ] Add my Lightning Callback plotting reconstructions/predictions every $N$ epochs

## Run tests

@TODO: Specify how to add tests and provide examples. But nobody likes testing.

```bash
# run all tests
pytest

# run all tests except the ones marked as slow
pytest -k "not slow"
```

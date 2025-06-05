# ---- Create missing datasets -----
python scripts/TSX_dataset_creation.py --input-dir data/TSX_cos_files --output-dir data/processed_hdf5/ --max-files 2 --norm-mode nat --norm-minmax 1 --clip --verbose

python scripts/TSX_dataset_creation.py --input-dir data/TSX_cos_files --output-dir data/processed_hdf5/ --max-files 2 --norm-mode nat --norm-minmax 5 --verbose

python scripts/TSX_dataset_creation.py --input-dir data/TSX_cos_files --output-dir data/processed_hdf5/ --max-files 2 --norm-mode nat --norm-minmax 5 --clip --verbose

python scripts/TSX_dataset_creation.py --input-dir data/TSX_cos_files --output-dir data/processed_hdf5/ --max-files 2 --norm-mode nat --norm-minmax 10 --clip --verbose

python scripts/TSX_dataset_creation.py --input-dir data/TSX_cos_files --output-dir data/processed_hdf5/ --max-files 2 --norm-mode nat --norm-minmax 5 --preserve-threshold 60 --verbose

# ---- All new datasets -----
python src/train.py experiment=sar_ddc model.criterion.lmbda=10 logger.wandb.run_name="sweep_pinten_no1%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm1natclip

python src/train.py experiment=sar_ddc model.criterion.lmbda=10 logger.wandb.run_name="sweep_pinten_no1%clip" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm1natclip

python src/train.py experiment=sar_ddc model.criterion.lmbda=10 logger.wandb.run_name="sweep_pinten_no5%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10nat

python src/train.py experiment=sar_ddc model.criterion.lmbda=10 logger.wandb.run_name="sweep_pinten_no5%clip" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm5natclip

python src/train.py experiment=sar_ddc model.criterion.lmbda=10 logger.wandb.run_name="sweep_pinten_no10%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10nat

python src/train.py experiment=sar_ddc model.criterion.lmbda=10 logger.wandb.run_name="sweep_pinten_no10%clip" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10natclip

# ---- A few lambdas -----
python src/train.py experiment=sar_ddc model.criterion.lmbda=0.001 logger.wandb.run_name="sweep0.001_pinten_no10%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10nat

python src/train.py experiment=sar_ddc model.criterion.lmbda=0.01 logger.wandb.run_name="sweep0.01_pinten_no10%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10nat

python src/train.py experiment=sar_ddc model.criterion.lmbda=0.1 logger.wandb.run_name="sweep0.1_pinten_no10%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10nat

python src/train.py experiment=sar_ddc model.criterion.lmbda=0.5 logger.wandb.run_name="sweep0.5_pinten_no10%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10nat

python src/train.py experiment=sar_ddc model.criterion.lmbda=1 logger.wandb.run_name="sweep1_pinten_no10%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10nat

python src/train.py experiment=sar_ddc model.criterion.lmbda=100 logger.wandb.run_name="sweep100_pinten_no10%" data.hdf5_dir=data/processed_hdf5/randomsplit2im_nopres_norm10nat
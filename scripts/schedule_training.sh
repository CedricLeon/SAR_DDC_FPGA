#!/bin/bash

# Run from root folder with: bash scripts/schedule_training.sh

# ----- Short schedule for training missing runs -----
python src/train.py -m experiment=ADAM seed=5 model.criterion.lmbda=1,2,5,10,20,50,100,200,500,1000 model.net.activation="relu" model.net.no_output_padding=False
python src/train.py -m experiment=ADAM seed=5 model.criterion.lmbda=1,2,5,10,20,50,100,200,500,1000 model.net.activation="gdn" model.net.no_output_padding=True,False

# # ----- Full schedule of results -----
# # Generate Ground-truth checkpoints
# python src/train.py experiment=MERLIN
# python src/train.py experiment=ADAM-NOC

# # Create datasets
# # @TODO: Get there paths to feed to create_dataset.py
# python scripts/create_dataset.py

# # Full training sweep ~13days
# python src/train.py -m experiment=ADAM seed=0,1,2,3,4 model.criterion.lmbda=1,2,5,10,20,50,100,200,500,1000 model.net.no_output_padding=True,False model.net.activation="gdn","relu"

# # Deployment on FPGA and evaluation
# # @TODO
# python master_deploy.py

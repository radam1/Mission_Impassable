"""
main.py: 1.0
This file is the main file for training and running the model.
this is a first test of git back and forth
"""

# Modules for Neural Network 
import torch
import torch.nn as nn 
import lightning as L
import torchvision
import torchvision.transforms as transforms

#h5py for handling data 
import h5py as h5 

# For Hydra Interface
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

#Instantiating the variables from Config/main-config.yaml

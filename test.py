"""
Testing script for DLF
"""
from run import DLF_run

DLF_run(model_name='DLF', dataset_name='mosei', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='test', is_training=False)

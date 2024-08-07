import yaml
import argparse
from argparse import Namespace
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument("--script", type=str,action='store',choices=['train','test','evaluation.compute_objective_metrics'],required=True, help='Path to script that needs to be ran.')
parser.add_argument("--config", type=str,action='store',required=True, help='Path to config file.')
args = parser.parse_args()


with open(args.config) as f:
    config_dict=yaml.safe_load(f)
    config_args=Namespace(**config_dict)

##import script
script=import_module(args.script)

script.main(config_args)

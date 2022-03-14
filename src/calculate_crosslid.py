import os
import argparse
from typing import List
import json
import yaml

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)

from src.crosslid.CrossLIDTest import CrossLID

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=List[str], default=["gan"], help="which models to evaluate. Will be overrided if 'models' is specified in params.yaml")
parser.add_argument("--results_location", type=str, default="results", help="where all models weights are stored")
parser.add_argument("--metrics_output", type=str, default="metrics.json", help="where to store metrics outputs")
opt = parser.parse_args()
print(opt)

metric = CrossLID()

cross_lid_scores = {}

models = params['models'] if params['models'] else opt.models

for model in models:
  weights_path = f'{opt.results_location}/{model}/weights/last.onnx'
  cross_lid_scores[f"{model}_crosslid"] = metric.calculate_cross_lid(model, weights_path)

with open('metrics.json', 'w+') as metrics_file:
  json.dump(cross_lid_scores, metrics_file)
import os
import argparse
from typing import List
import json

from src.crosslid.CrossLIDTest import CrossLID

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=List, default=["gan"], help="which models to evaluate. Options are 'gan', 'cgan' or 'dcgan'")
parser.add_argument("--results_location", type=str, default="results", help="where all models weights are stored")
parser.add_argument("--metrics_output", type=str, default="metrics.json", help="where to store metrics outputs")
opt = parser.parse_args()
print(opt)

metric = CrossLID()

cross_lid_scores = {}

for model in opt.models:
  weights_path = f'{opt.results_location}/{model}/weights/last.onnx'
  cross_lid_scores[f"{model}_crosslid"] = metric.calculate_cross_lid(weights_path)

with open('metrics.json', 'w+') as metrics_file:
  json.dump(cross_lid_scores, metrics_file)
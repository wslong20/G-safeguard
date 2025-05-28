import json
import os
import argparse

parser = argparse.ArgumentParser(description="Experiments that generate dataset")
parser.add_argument("--dataset", type=str, default="mmlu")
parser.add_argument("--phase", type=str, default="train")
args = parser.parse_args()

if args.dataset == "mmlu": 
    args.root = "./agent_graph_dataset/mmlu"
elif args.dataset == "csqa": 
    args.root = "./agent_graph_dataset/csqa"
elif args.dataset == "gsm8k": 
    args.root = "./agent_graph_dataset/gsm8k"
else: 
    raise Exception(f"Unknown dataset {args.dataset}")

root = os.path.join(args.root, args.phase)
files = os.listdir(root)

datafiles = []
for file in files: 
    data_file = os.path.join(root, file)
    datafiles.append(data_file)

dataset = []
for datafile in datafiles: 
    with open(datafile, "r") as f: 
        data = json.load(f)
    dataset += data

save_file = os.path.join(root, "dataset.json")
with open(save_file, "w") as f:
    json.dump(dataset, f, indent=None)


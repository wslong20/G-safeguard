# G-Safeguard

## Overview

We provide the code of our paper "G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems". 
## Quick Start

### Install packages

```bash
conda create -n gsafeguard python=3.10
conda activate gsafeguard
pip install -r requirements.txt
```

### Add API keys in `template.env` and change its name to `.env`

```python
BASE_URL = "" 
API_KEY = ""
```

### For memory attack, run the following command to execute G-safeguard

```bash
cd MA
python gen_graph.py --num_nodes 8 --sparsity 0.5 --num_graphs 50 --num_attackers 3 --samples 800 --model_type gpt-4o-mini
python gen_training_dataset.py
python train.py
python main_defense_for_different_topology.py  --graph_type random --gnn_checkpoint_path {gnn_model_save_path} --model_type {gpt-4o-mini}
```
{gnn_model_save_path} needs to be replaced with the save path of the gnn model, which can be viewed in train.py. --model_type can be set to other models besides gpt-4o-mini, such as gpt-4o, llama 3.1-70b, etc., but your api backend must support other models

### Other attack Settings, same as above. However, for Prompt Injection, it may be necessary to change the meta_dataset in gen_training_set.py to match the dataset being used
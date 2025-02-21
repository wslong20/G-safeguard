# G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems

## 📰 News

🚩 Updates (2025-2-26) Initial upload to arXiv [PDF](https://arxiv.org/abs/2502.11127).

## 🌟 Overview

We introduce **G-Safeguard**, the first security safeguard for LLM-based multi-agent systems. It is a topology-guided security lens and treatment for robust LLM-MAS, which leverages graph neural networks to detect anomalies on the multi-agent utterance graph and employ topological intervention for attack remediation. 

![](./assets/framework-1.pdf)

## 🛠 Quick Start

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

### 💻 Memory Attack

```bash
cd MA
python gen_graph.py --num_nodes 8 --sparsity 0.5 --num_graphs 50 --num_attackers 3 --samples 800 --model_type gpt-4o-mini
python gen_training_dataset.py
python train.py
python main_defense_for_different_topology.py  --graph_type random --gnn_checkpoint_path {gnn_model_save_path} --model_type {gpt-4o-mini}
```
`{gnn_model_save_path}` needs to be replaced with the save path of the gnn model, which can be viewed in train.py. --model_type can be set to other models besides gpt-4o-mini, such as gpt-4o, llama 3.1-70b, etc., but your api backend must support other models

---

Note: Other attack settings follow similar commands. However, for Prompt Injection, it may be necessary to change the meta_dataset in gen_training_set.py to match the dataset being used



## 📚 Citation
If you find this repo useful, please consider citing our paper as follows:
```bibtex
@article{wang2025g-safeguard,
  title={G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems},
  author={Wang, Shilong and Zhang, Guibin and Yu, Miao and Wan, Guancheng and Meng, Fanci and Guo, Chongye and Wang, Kun and Wang, Yang},
  journal={arXiv preprint arXiv:2502.11127},
  year={2025}
}

@article{zhang2024agentprune,
  title={Cut the crap: An economical communication pipeline for llm-based multi-agent systems},
  author={Zhang, Guibin and Yue, Yanwei and Li, Zhixun and Yun, Sukwon and Wan, Guancheng and Wang, Kun and Cheng, Dawei and Yu, Jeffrey Xu and Chen, Tianlong},
  journal={arXiv preprint arXiv:2410.02506},
  year={2024}
}
```

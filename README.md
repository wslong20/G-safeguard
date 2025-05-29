# [ACL 2025 Main] G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems

## 📰 News

- 🎉 Updates (2025-5-15) G-Safeguard is accepted to ACL 2025 Main!
- 🚩 Updates (2025-2-26) Initial upload to arXiv [PDF](https://arxiv.org/abs/2502.11127).

## 🌟 Overview

We introduce **G-Safeguard**, the first security safeguard for LLM-based multi-agent systems. It is a topology-guided security lens and treatment for robust LLM-MAS, which leverages graph neural networks to detect anomalies on the multi-agent utterance graph and employ topological intervention for attack remediation. 

![](./assets/pipeline.png)

## 🛠 Quick Start

### Install packages

```bash
conda create -n gsafeguard python=3.10
conda activate gsafeguard
pip install -r requirements.txt
```

### Set Environment Variables

```bash
export BASE_URL=""
export OPENAI_API_KEY=""
```

### Memory Attack
Please refer to [`Memory Attack README.md`](./MA/README.md).
### Prompt Injection
Please refer to [`Prompt Injection README.md`](./PI/README.md).
### For training G-safeguard for Tool Attack, please refer to:
Please refer to [`Tool Attack README.md`](./TA/README.md).
### Scalability Experiment
Please refer to [`Scalability README.md`](./scalability/README.md).


## 📚 Citation
If you find this repo useful, please consider citing our paper as follows:
```bibtex
@article{wang2025g-safeguard,
  title={G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems},
  author={Wang, Shilong and Zhang, Guibin and Yu, Miao and Wan, Guancheng and Meng, Fanci and Guo, Chongye and Wang, Kun and Wang, Yang},
  journal={arXiv preprint arXiv:2502.11127},
  year={2025}
}

@article{zhang2024g-designer,
  title={G-designer: Architecting multi-agent communication topologies via graph neural networks},
  author={Zhang, Guibin and Yue, Yanwei and Sun, Xiangguo and Wan, Guancheng and Yu, Miao and Fang, Junfeng and Wang, Kun and Chen, Tianlong and Cheng, Dawei},
  journal={arXiv preprint arXiv:2410.11782},
  year={2024}
}

@article{zhang2024agentprune,
  title={Cut the crap: An economical communication pipeline for llm-based multi-agent systems},
  author={Zhang, Guibin and Yue, Yanwei and Li, Zhixun and Yun, Sukwon and Wan, Guancheng and Wang, Kun and Cheng, Dawei and Yu, Jeffrey Xu and Chen, Tianlong},
  journal={arXiv preprint arXiv:2410.02506},
  year={2024}
}
```

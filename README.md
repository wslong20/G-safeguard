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

### Set Environment Variables

```bash
export BASE_URL=""
export OPENAI_API_KEY=""
```

### For training G-safeguard for memory attack, please refer to:
[Memory Attack README.md](./MA/README.md)
### For training G-safeguard for Prompt Injection, please refer to:
[Prompt Injection README.md](./PI/README.md)
### For training G-safeguard for Tool Attack, please refer to:
[Tool Attack README.md](./TA/README.md)
### For experiments on the scalability of G-safeguard, please refer to:
[Scalability README.md](./scalability/README.md)
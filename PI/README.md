## Prompt Injection

### MMLU

```bash
cd PI

chmod +x ./scripts/train/gen_conversation_train_mmlu.sh
chmod +x ./scripts/test/gen_conversation_test_mmlu.sh

# Generate communication datasets for the training and testing phases
.\scripts\train\gen_conversation_train_mmlu.sh && python merge_datasets.py --phase train --dataset mmlu
.\scripts\train\gen_conversation_test_mmlu.sh && python merge_datasets.py --phase test --dataset mmlu
# Generate the dataset for training GNNs
python gen_training_dataset.py --dataset mmlu

# Train GNN
python train.py --dataset mmlu --epochs 50 --batch_size 32 --lr 0.001

# G-Safeguard test
python main_defense_for_different_topology.py  --graph_type random --gnn_checkpoint_path {gnn_model_save_path} --model_type gpt-4o-mini --dataset mmlu --samples 60
```
##### {gnn_model_save_path} needs to be replaced with the save path of the GNN. After GNN training, it is saved by default in "./checkpoint". model_type can be set to other models besides gpt-4o-mini, such as gpt-4o, llama 3.1-70b, etc., but your api backend must support these models


### CSQA
```bash
cd PI

chmod +x ./scripts/train/gen_conversation_train_csqa.sh
chmod +x ./scripts/test/gen_conversation_test_csqa.sh

# Generate communication datasets for the training and testing phases
.\scripts\train\gen_conversation_train_csqa.sh && python merge_datasets.py --phase train --dataset csqa
.\scripts\train\gen_conversation_test_csqa.sh && python merge_datasets.py --phase test --dataset csqa
# Generate the dataset for training GNNs
python gen_training_dataset.py --dataset csqa

# Train GNN
python train.py --dataset csqa --epochs 50 --batch_size 32 --lr 0.001

# G-Safeguard test
python main_defense_for_different_topology.py  --graph_type random --gnn_checkpoint_path {gnn_model_save_path} --model_type gpt-4o-mini --dataset csqa --samples 60
```
##### {gnn_model_save_path} needs to be replaced with the save path of the GNN. After GNN training, it is saved by default in "./checkpoint". model_type can be set to other models besides gpt-4o-mini, such as gpt-4o, llama 3.1-70b, etc., but your api backend must support these models


### GSM8K

```bash
cd PI

chmod +x ./scripts/train/gen_conversation_train_gsm8k.sh
chmod +x ./scripts/test/gen_conversation_test_gsm8k.sh

# Generate communication datasets for the training and testing phases
.\scripts\train\gen_conversation_train_gsm8k.sh && python merge_datasets.py --phase train --dataset gsm8k
.\scripts\train\gen_conversation_test_gsm8k.sh && python merge_datasets.py --phase test --dataset gsm8k
# Generate the dataset for training GNNs
python gen_training_dataset.py --dataset gsm8k

# Train GNN
python train.py --dataset gsm8k --epochs 50 --batch_size 32 --lr 0.001

# G-Safeguard test
python main_defense_for_different_topology.py  --graph_type random --gnn_checkpoint_path {gnn_model_save_path} --model_type gpt-4o-mini --dataset gsm8k --samples 60
```
##### {gnn_model_save_path} needs to be replaced with the save path of the GNN. After GNN training, it is saved by default in "./checkpoint". model_type can be set to other models besides gpt-4o-mini, such as gpt-4o, llama 3.1-70b, etc., but your api backend must support these models
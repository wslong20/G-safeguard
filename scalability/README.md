## Scalability

```bash
cd scalability

chmod +x ./scripts/train/gen_conversation_train.sh
chmod +x ./scripts/test/gen_conversation_test.sh

# Generate communication datasets for the training and testing phases
.\scripts\train\gen_conversation_train.sh && .\scripts\train\merge_datasets.sh
.\scripts\train\gen_conversation_test.sh && .\scripts\test\merge_datasets.sh
# Generate the dataset for training GNNs
python gen_training_dataset.py

# Train GNN
python train.py --epochs 50 --batch_size 32 --lr 0.001

# G-Safeguard test
python main_defense_for_different_topology.py  --graph_type random --gnn_checkpoint_path {gnn_model_save_path} --model_type gpt-4o-mini
```
###### {gnn_model_save_path} needs to be replaced with the save path of the GNN. After GNN training, it is saved by default in "./checkpoint". model_type can be set to other models besides gpt-4o-mini, such as gpt-4o, llama 3.1-70b, etc., but your api backend must support other models

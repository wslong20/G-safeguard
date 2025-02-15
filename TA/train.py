import argparse
import os
from tqdm import tqdm
from data import AgentGraphDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import MyGAT
from einops import rearrange
from datetime import datetime
import random 

def train(model: MyGAT, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        x = edge_attr[:, 0, :]
        x = scatter_mean(x, edge_index[0], dim=0, dim_size=len(data.x))
        random_turns = random.choice(list(range(1, 5)))
        edge_attr[:, :random_turns, :]
        optimizer.zero_grad()
        outputs = model(x, edge_index=edge_index, edge_attr=edge_attr)
        loss = criterion(outputs, y.float().unsqueeze(-1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
        total += y.size(0)
        correct += (predicted == y).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)
            x = edge_attr[:, 0, :]
            x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(data.x))

            outputs = model(x, edge_index, edge_attr)
            loss = criterion(outputs, y.float().unsqueeze(-1))
            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
            total += y.size(0)
            correct += (predicted == y).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")

    parser.add_argument("--dataset_path", type=str, default="./ModelTrainingSet/tool_attack/dataset.pkl", help="Save path of the dataset")
    
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--save_dir", type=str, default="./checkpoint")


    args = parser.parse_args()

    normalized_path = os.path.normpath(args.dataset_path)
    parts = normalized_path.split(os.sep)
    dataset = parts[-2]
    args.save_dir = os.path.join(args.save_dir, dataset)

    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time_str}-hiddim_{args.hidden_dim}-heads_{args.num_heads}-layers_{args.num_layers}-epochs_{args.epochs}-lr_{args.lr}-dropout_{args.dropout}-wd_{args.weight_decay}.pth"
    args.save_path = os.path.join(args.save_dir, filename)

    return args


def main():
    args = parse_arguments()
    
    train_dataset = AgentGraphDataset(args.dataset_path, phase="train")
    val_dataset = AgentGraphDataset(args.dataset_path, phase="val")
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(val_dataset)
    example = train_dataset[0]
    in_channels = example.x.size(1)
    edge_dim = example.edge_attr.size()[1:]

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    gnn = MyGAT(in_channels, args.hidden_dim, out_channels=1, heads=args.num_heads, num_layers=args.num_layers, edge_dim=edge_dim)
    gnn.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best_acc = 0.0

    for i in range(args.epochs): 
        train_loss, train_acc = train(gnn, trainloader, criterion, optimizer, device=device)
        test_loss, test_acc = test(gnn, testloader, criterion, device=device)
        scheduler.step()
        if test_acc > best_acc: 
            best_acc = test_acc
            torch.save(gnn.state_dict(), args.save_path)  # 保存模型
            print(f"Epoch {i}/{args.epochs} || Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}% || Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}% || Save!")
        else:
            print(f"Epoch {i}/{args.epochs} || Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}% || Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()




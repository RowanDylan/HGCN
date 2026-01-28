import argparse
import os
import torch
import random
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, Linear
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score, f1_score, recall_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def split_dataset(geo_chem, y, seed=42):
    y_np = y.cpu().numpy()
    label_1_idx = np.where(y_np == 1)[0]
    label_0_idx = np.where(y_np == 0)[0]
    label_0_data = geo_chem.iloc[label_0_idx]
    label_0_original_idx = label_0_data.sort_values(
        by=['rock_oushi', 'fault_oush', 'Cu'],
        ascending=[False, False, True]
    ).head(25000).index.values
    train_idx_1, val_idx_1 = train_test_split(label_1_idx, test_size=0.2, random_state=seed)
    train_idx_0, val_idx_0 = train_test_split(label_0_original_idx, test_size=0.2, random_state=seed)
    train_idx = np.concatenate([train_idx_1, train_idx_0])
    val_idx = np.concatenate([val_idx_1, val_idx_0])
    test_idx = np.arange(len(y))
    return train_idx, val_idx, test_idx

def load_adj_matrix(file_name, num_nodes):
    df = pd.read_csv(os.path.join(data_dir, file_name))
    edge_index = torch.tensor(df.values, dtype=torch.long).t().contiguous()
    edge_index = torch.clamp(edge_index, 0, num_nodes - 1)
    return edge_index

def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

class HGNN(torch.nn.Module):
    def __init__(self, input_dim=-1, hidden_dim=32, output_dim=2, dropout=0.2):
        super().__init__()
        self.conv1 = HeteroConv({
            ('geochem', 'fault_density', 'geochem'): GCNConv(input_dim, hidden_dim),
            ('geochem', 'fault', 'geochem'): GCNConv(input_dim, hidden_dim),
            ('geochem', 'rock', 'geochem'): GCNConv(input_dim, hidden_dim),
            ('geochem', 'geo', 'geochem'): GCNConv(input_dim, hidden_dim),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('geochem', 'fault_density', 'geochem'): GCNConv(hidden_dim, hidden_dim),
            ('geochem', 'fault', 'geochem'): GCNConv(hidden_dim, hidden_dim),
            ('geochem', 'rock', 'geochem'): GCNConv(hidden_dim, hidden_dim),
            ('geochem', 'geo', 'geochem'): GCNConv(hidden_dim, hidden_dim),
        }, aggr='sum')
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x = x_dict['geochem']
        x = self.bn(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x

def train_model(model, data, optimizer, criterion, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_aucs = []
    val_aucs = []
    test_aucs = []
    train_recalls = []
    val_recalls = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        train_loss = criterion(out[data['geochem'].train_mask], data['geochem'].y[data['geochem'].train_mask])
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        train_probs = F.softmax(out[data['geochem'].train_mask], dim=1)[:, 1]
        train_pred = (train_probs > 0.5).float().long()
        train_accuracy = (train_pred == data['geochem'].y[data['geochem'].train_mask]).float().mean().item()
        train_accuracies.append(train_accuracy)

        train_probs_np = train_probs.detach().cpu().numpy()
        train_labels = data['geochem'].y[data['geochem'].train_mask].cpu().numpy()
        try:
            train_auc = roc_auc_score(train_labels, train_probs_np)
        except ValueError:
            train_auc = 0.5
        train_aucs.append(train_auc)

        train_recall = recall_score(train_labels, train_pred.cpu().numpy())
        train_recalls.append(train_recall)

        model.eval()
        with torch.no_grad():
            out = model(data)
            val_loss = criterion(out[data['geochem'].val_mask], data['geochem'].y[data['geochem'].val_mask])
            val_losses.append(val_loss.item())

            val_probs = F.softmax(out[data['geochem'].val_mask], dim=1)[:, 1]
            val_pred = (val_probs > 0.5).float().long()
            val_accuracy = (val_pred == data['geochem'].y[data['geochem'].val_mask]).float().mean().item()
            val_accuracies.append(val_accuracy)

            val_probs_np = val_probs.detach().cpu().numpy()
            val_labels = data['geochem'].y[data['geochem'].val_mask].cpu().numpy()
            try:
                val_auc = roc_auc_score(val_labels, val_probs_np)
            except ValueError:
                val_auc = 0.5
            val_aucs.append(val_auc)

            test_probs = F.softmax(out[data['geochem'].test_mask], dim=1)[:, 1]
            test_probs_np = test_probs.detach().cpu().numpy()
            test_labels = data['geochem'].y[data['geochem'].test_mask].cpu().numpy()
            try:
                test_auc = roc_auc_score(test_labels, test_probs_np)
            except ValueError:
                test_auc = 0.5
            test_aucs.append(test_auc)

            val_recall = recall_score(val_labels, val_pred.cpu().numpy())
            val_recalls.append(val_recall)

        print(f'Epoch: {epoch:03d}')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training AUC: {train_auc:.4f}, Training Recall: {train_recall:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}, Validation Recall: {val_recall:.4f}')
        print(f'Test AUC: {test_auc:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies, train_aucs, val_aucs, test_aucs, train_recalls, val_recalls

def test_model(model, data, criterion, device):
    model.eval()
    with torch.no_grad():
        out = model(data)
        test_loss = criterion(out[data['geochem'].test_mask], data['geochem'].y[data['geochem'].test_mask])
        test_probs = F.softmax(out[data['geochem'].test_mask], dim=1)[:, 1]
        test_probs_np = test_probs.detach().cpu().numpy()
        test_labels = data['geochem'].y[data['geochem'].test_mask].cpu().numpy()
        test_pred = (test_probs > 0.5).float().long()
        test_accuracy = (test_pred == data['geochem'].y[data['geochem'].test_mask]).float().mean().item()
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        fpr, tpr, thresholds = roc_curve(test_labels, test_probs_np)
        test_auc = roc_auc_score(test_labels, test_probs_np)

    return fpr, tpr, test_auc

def calculate_metrics(model, data, y):
    model.eval()
    with torch.no_grad():
        out = model(data)
        train_y = y[data['geochem'].train_mask.cpu().numpy()]
        val_y = y[data['geochem'].val_mask.cpu().numpy()]

        train_probs = F.softmax(out[data['geochem'].train_mask], dim=1)[:, 1]
        train_pred = (train_probs > 0.5).float().long().cpu().numpy()
        train_true = train_y.cpu().numpy()

        train_kappa = cohen_kappa_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred)
        train_recall = recall_score(train_true, train_pred)

        val_probs = F.softmax(out[data['geochem'].val_mask], dim=1)[:, 1]
        val_pred = (val_probs > 0.5).float().long().cpu().numpy()
        val_true = val_y.cpu().numpy()

        val_kappa = cohen_kappa_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred)
        val_recall = recall_score(val_true, val_pred)

    return train_kappa, train_f1, train_recall, val_kappa, val_f1, val_recall

def plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies, test_aucs, fpr, tpr, test_auc, train_cm, val_cm, train_recalls, val_recalls, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_accuracy.png'))
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(range(len(test_aucs)), test_aucs, label='Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Test AUC over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_auc.png'))
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, 'r-', label=f'HGCN* (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Training Set Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.subplot(1, 2, 2)
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Validation Set Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(range(len(train_recalls)), train_recalls, label='Training Recall')
    plt.plot(range(len(val_recalls)), val_recalls, label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'recall.png'))
    plt.show()

def calculate_confusion_matrix(model, data, y):
    model.eval()
    with torch.no_grad():
        out = model(data)
        train_y = y[data['geochem'].train_mask.cpu().numpy()]
        val_y = y[data['geochem'].val_mask.cpu().numpy()]

        train_probs = F.softmax(out[data['geochem'].train_mask], dim=1)[:, 1]
        train_pred = (train_probs > 0.5).float().long().cpu().numpy()
        train_true = train_y.cpu().numpy()
        train_cm = confusion_matrix(train_true, train_pred)

        val_probs = F.softmax(out[data['geochem'].val_mask], dim=1)[:, 1]
        val_pred = (val_probs > 0.5).float().long().cpu().numpy()
        val_true = val_y.cpu().numpy()
        val_cm = confusion_matrix(val_true, val_pred)

    return train_cm, val_cm

def main():
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--data-dir", default=os.path.join(script_dir, "data"))
    parser.add_argument("--out-dir", default=os.path.join(script_dir, "outputs"))
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    global data_dir
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)

    set_seed(args.seed)

    require_file(os.path.join(data_dir, "data.csv"))
    require_file(os.path.join(data_dir, "fault_density_adj_matrix.csv"))
    require_file(os.path.join(data_dir, "fault_adj_matrix.csv"))
    require_file(os.path.join(data_dir, "geo_adj_matrix.csv"))
    require_file(os.path.join(data_dir, "rock_adj_matrix.csv"))

    geochem_data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
    features = geochem_data.drop(columns=['Ore']).iloc[:, 2:-5]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    node_ids = geochem_data.iloc[:, 0].values
    y = torch.tensor(geochem_data['Ore'].values, dtype=torch.long)

    train_idx, val_idx, test_idx = split_dataset(geochem_data, y, seed=args.seed)

    y_np = y.cpu().numpy()
    train_y = y_np[train_idx]
    val_y = y_np[val_idx]
    test_y = y_np[test_idx]

    train_positive = (train_y == 1).sum()
    train_negative = (train_y == 0).sum()
    val_positive = (val_y == 1).sum()
    val_negative = (val_y == 0).sum()
    test_positive = (test_y == 1).sum()
    test_negative = (test_y == 0).sum()

    data = HeteroData()
    data['geochem'].x = torch.tensor(scaled_features, dtype=torch.float)
    data['geochem'].y = y

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data['geochem'].train_mask = train_mask
    data['geochem'].val_mask = val_mask
    data['geochem'].test_mask = test_mask

    num_nodes = data['geochem'].x.size(0)

    edge_index_fault_density = load_adj_matrix('fault_density_adj_matrix.csv', num_nodes)
    edge_index_fault = load_adj_matrix('fault_adj_matrix.csv', num_nodes)
    edge_index_geo = load_adj_matrix('geo_adj_matrix.csv', num_nodes)
    edge_index_rock = load_adj_matrix('rock_adj_matrix.csv', num_nodes)

    edge_index_list = [edge_index_fault_density, edge_index_fault, edge_index_geo, edge_index_rock]
    for edge_index in edge_index_list:
        assert edge_index.max() < num_nodes, f"edge_index error: {edge_index.max()}"

    data['geochem', 'fault_density', 'geochem'].edge_index = edge_index_fault_density
    data['geochem', 'fault', 'geochem'].edge_index = edge_index_fault
    data['geochem', 'geo', 'geochem'].edge_index = edge_index_geo
    data['geochem', 'rock', 'geochem'].edge_index = edge_index_rock

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = HGNN().to(device)
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = args.epochs

    result_path = os.path.join(out_dir, "HGNN_")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder_name = f"{current_time}_epochs_{num_epochs}_lr_{lr}"
    save_path = os.path.join(result_path, save_folder_name)
    os.makedirs(save_path, exist_ok=True)

    data = data.to(device)
    train_losses, val_losses, train_accuracies, val_accuracies, train_aucs, val_aucs, test_aucs, train_recalls, val_recalls = train_model(
        model, data, optimizer, criterion, num_epochs, device
    )

    fpr, tpr, test_auc = test_model(model, data, criterion, device)

    train_kappa, train_f1, train_recall, val_kappa, val_f1, val_recall = calculate_metrics(model, data, y)
    print(f"Train Kappa: {train_kappa:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}")
    print(f"Validation Kappa: {val_kappa:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}")

    train_cm, val_cm = calculate_confusion_matrix(model, data, y)
    print("Train confusion matrix:")
    print(train_cm)
    print("Validation confusion matrix:")
    print(val_cm)

    metrics_df = pd.DataFrame({
        'train_kappa': [train_kappa],
        'train_f1': [train_f1],
        'train_recall': [train_recall],
        'val_kappa': [val_kappa],
        'val_f1': [val_f1],
        'val_recall': [val_recall],
        'train_confusion_matrix': [str(train_cm)],
        'val_confusion_matrix': [str(val_cm)]
    })
    metrics_df.to_csv(os.path.join(save_path, 'evaluation_metrics.csv'), index=False)

    model.eval()
    with torch.no_grad():
        out = model(data)
        test_probs = F.softmax(out[data['geochem'].test_mask], dim=1)[:, 1].detach().cpu().numpy()

    test_xx_yy = geochem_data.loc[test_idx, ['XX', 'YY']].reset_index(drop=True)

    test_results = pd.DataFrame({
        'oreq': test_probs
    })
    test_results = pd.concat([test_results, test_xx_yy], axis=1)

    test_file_name = f"{current_time}_epochs_{num_epochs}_lr_{lr}_test_prediction_probabilities.csv"
    test_results.to_csv(os.path.join(save_path, test_file_name), index=False)

    roc_results_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr
    })
    roc_file_name = f"HGNN_{current_time}_epochs_{num_epochs}_lr_{lr}_test_roc.csv"
    roc_results_df.to_csv(os.path.join(save_path, roc_file_name), index=False)

    plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies, test_aucs, fpr, tpr, test_auc, train_cm, val_cm, train_recalls, val_recalls, save_path)

if __name__ == "__main__":
    main()

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
from sklearn.preprocessing import MinMaxScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dense = None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        if self.dense is None:
            input_features = x.shape[1]
            self.dense = nn.Linear(input_features, 2)

        x = self.dense(x)
        return x

def calculate_metrics(labels, predicted):
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted)
    recall = recall_score(labels, predicted)
    f1 = f1_score(labels, predicted)
    conf_matrix = confusion_matrix(labels, predicted)
    mcc = matthews_corrcoef(labels, predicted)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    kappa = cohen_kappa_score(labels, predicted)
    balanced_accuracy = balanced_accuracy_score(labels, predicted)
    return accuracy, precision, recall, f1, mcc, specificity, conf_matrix, kappa, balanced_accuracy

def train_and_evaluate(lr, save_dir, num_epochs, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    train_losses = []
    train_metrics = []
    train_aucs = []
    train_accuracies = []
    val_losses = []
    val_metrics = []
    val_aucs = []
    val_accuracies = []
    test_aucs = []

    model = CNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_batch_size = len(X_train)
    val_batch_size = len(X_val)
    test_batch_size = len(X_test)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_probs = []
        train_labels = []
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            train_probs.extend(probabilities.detach().cpu().numpy().tolist())
            train_labels.extend(labels.detach().cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_running_loss = 0.0
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1)
                labels = labels.long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probabilities.detach().cpu().numpy().tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())

        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        train_predicted = [1 if prob > 0.5 else 0 for prob in train_probs]
        train_accuracy, train_precision, train_recall, train_f1, train_mcc, train_specificity, train_conf_matrix, train_kappa, train_balanced_accuracy = calculate_metrics(train_labels, train_predicted)
        train_metrics.append([train_accuracy, train_precision, train_recall, train_f1, train_mcc, train_specificity, train_kappa, train_balanced_accuracy])
        train_accuracies.append(train_accuracy)

        val_predicted = [1 if prob > 0.5 else 0 for prob in val_probs]
        val_accuracy, val_precision, val_recall, val_f1, val_mcc, val_specificity, val_conf_matrix, val_kappa, val_balanced_accuracy = calculate_metrics(val_labels, val_predicted)
        val_metrics.append([val_accuracy, val_precision, val_recall, val_f1, val_mcc, val_specificity, val_kappa, val_balanced_accuracy])
        val_accuracies.append(val_accuracy)

        from sklearn.metrics import roc_curve, auc
        train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
        train_auc = auc(train_fpr, train_tpr)
        train_aucs.append(train_auc)

        val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)
        val_auc = auc(val_fpr, val_tpr)
        val_aucs.append(val_auc)

        model.eval()
        all_labels = []
        all_predicted = []
        test_probs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)
                labels = labels.long()

                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                test_probs.extend(probabilities.detach().cpu().numpy().tolist())
                predicted = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.detach().cpu().numpy().tolist())
                all_predicted.extend(predicted.detach().cpu().numpy().tolist())

        test_fpr, test_tpr, _ = roc_curve(all_labels, test_probs)
        test_auc = auc(test_fpr, test_tpr)
        test_aucs.append(test_auc)

    datasets = ['Train', 'Validation']
    conf_matrices = [train_conf_matrix, val_conf_matrix]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (dataset, conf_matrix) in enumerate(zip(datasets, conf_matrices)):
        ax = axes[i]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{dataset} Set Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path_confusion = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path_confusion)
    plt.show()

    train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
    train_auc = auc(train_fpr, train_tpr)

    val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)
    val_auc = auc(val_fpr, val_tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(train_fpr, train_tpr, label=f'Train ROC curve (area = {train_auc:.2f})')
    plt.plot(val_fpr, val_tpr, label=f'Validation ROC curve (area = {val_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for Train, Validation sets')
    plt.legend(loc="lower right")
    save_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(save_path)
    plt.show()

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Specificity', 'Kappa', 'Balanced Accuracy', 'AUC']
    datasets = ['Train', 'Validation']
    train_final_auc = train_aucs[-1]
    val_final_auc = val_aucs[-1]

    train_metrics[-1].append(train_final_auc)
    val_metrics[-1].append(val_final_auc)

    metrics_values = [train_metrics[-1], val_metrics[-1]]

    for i, dataset in enumerate(datasets):
        for j, metric_name in enumerate(metrics_names):
            print(f"{metric_name}: {metrics_values[i][j]:.4f}")

    metrics_df = pd.DataFrame(metrics_values, index=datasets, columns=metrics_names)
    metrics_csv_path = os.path.join(save_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    save_path_loss_acc = os.path.join(save_dir, 'loss_accuracy_curves.png')
    plt.savefig(save_path_loss_acc)

    plt.figure(figsize=(10, 8))
    plt.plot(range(num_epochs), train_aucs, label='Train AUC')
    plt.plot(range(num_epochs), val_aucs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC vs Epoch')
    plt.legend()
    save_path_auc = os.path.join(save_dir, 'auc_vs_epoch.png')
    plt.savefig(save_path_auc)

    train_recalls = [metric[2] for metric in train_metrics]
    val_recalls = [metric[2] for metric in val_metrics]

    plt.figure(figsize=(6, 5))
    plt.plot(range(num_epochs), train_recalls, label='Train Recall')
    plt.plot(range(num_epochs), val_recalls, label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall over Epochs')
    plt.legend()
    plt.tight_layout()
    save_path_recall = os.path.join(save_dir, 'recall_vs_epoch.png')
    plt.savefig(save_path_recall)

    plt.figure(figsize=(10, 8))
    plt.plot(range(num_epochs), test_aucs, label='Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Test AUC vs Epoch')
    plt.legend()
    save_path_test_auc = os.path.join(save_dir, 'test_auc_vs_epoch.png')
    plt.savefig(save_path_test_auc)

    RUN_PREDICTION_SAVE = True
    if RUN_PREDICTION_SAVE:
        model.eval()
        test_probs = []
        test_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)[:, 1]
                test_probs.extend(probabilities.cpu().tolist())
                test_labels.extend(labels.cpu().tolist())

    test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
    test_auc = auc(test_fpr, test_tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(test_fpr, test_tpr, label=f'CNN (AUC = {test_auc:.4f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Test set')
    plt.legend(loc="lower right")

    save_path_test_roc = os.path.join(save_dir, 'test_roc_curve.png')
    plt.savefig(save_path_test_roc)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    roc_results_df = pd.DataFrame({
        'fpr': test_fpr,
        'tpr': test_tpr
    })
    roc_file_name = f"CNN_{current_time}_epochs_{num_epochs}_lr_{lr}_test_roc.csv"
    roc_results_df.to_csv(os.path.join(save_dir, roc_file_name), index=False)

    last_two_columns = data.iloc[:, -4:-2]

    result_df = pd.DataFrame({
        'Predicted_Probability': test_probs
    })

    final_result = pd.concat([result_df, last_two_columns], axis=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file_path = os.path.join(save_dir, f'prediction_results_{timestamp}.csv')
    final_result.to_csv(save_file_path, index=False)

    plt.close('all')
    return model, val_metrics[-1][2]

def custom_split_samples(X, y, coords, data, seed=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    original_positive_mask = y == 1

    bottom_25000 = data.sort_values(by=['rock_oushi', 'fault_oush', 'Cu'],
                                     ascending=[False, False, True]).head(25000)
    negative_indices = bottom_25000.index

    X_positive = X[np.where(original_positive_mask)[0]]
    y_positive = y[np.where(original_positive_mask)[0]]
    X_negative = X[negative_indices]
    y_negative = np.zeros(len(negative_indices))

    X_combined = np.concatenate([X_positive, X_negative])
    y_combined = np.concatenate([y_positive, y_negative])

    positive_indices = np.where(y_combined == 1)[0]
    negative_indices = np.where(y_combined == 0)[0]

    X_positive = X_combined[positive_indices]
    y_positive = y_combined[positive_indices]

    X_positive_train, X_positive_val, y_positive_train, y_positive_val = train_test_split(
        X_positive, y_positive, test_size=0.2, random_state=seed
    )

    X_negative = X_combined[negative_indices]
    y_negative = y_combined[negative_indices]

    X_negative_train, X_negative_val, y_negative_train, y_negative_val = train_test_split(
        X_negative, y_negative, test_size=0.2, random_state=seed
    )

    X_train = np.concatenate([X_positive_train, X_negative_train])
    y_train = np.concatenate([y_positive_train, y_negative_train])

    X_val = np.concatenate([X_positive_val, X_negative_val])
    y_val = np.concatenate([y_positive_val, y_negative_val])

    X_test = X
    y_test = y

    train_pos = np.sum(y_train == 1)
    train_neg = np.sum(y_train == 0)
    val_pos = np.sum(y_val == 1)
    val_neg = np.sum(y_val == 0)
    test_pos = np.sum(y_test == 1)
    test_neg = np.sum(y_test == 0)

    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_feature_distribution(X, feature_names, dataset_name):
    df = pd.DataFrame(X, columns=feature_names)
    df.hist(bins=20, figsize=(15, 10))
    plt.suptitle(f'{dataset_name} feature distribution')
    plt.show()

def plot_label_distribution(y, dataset_name):
    positive_count = np.sum(y == 1)
    negative_count = np.sum(y == 0)
    label_dist = pd.DataFrame({
        'Sample Class': ['Positive Samples', 'Negative Samples'],
        'Count': [positive_count, negative_count]
    })

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x='Sample Class', y='Count', data=label_dist)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

    plt.title(f'{dataset_name} Positive and Negative Sample Distribution')
    plt.xlabel('Sample Class')
    plt.ylabel('Count')
    plt.show()

def feature_importance(model, X, y, feature_names, save_dir, n_repeats=5):
    model.eval()
    inputs = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(y, dtype=torch.long)
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1)
    baseline_accuracy = (preds == labels).sum().item() / len(labels)

    importance_scores = []
    for i in range(X.shape[1]):
        feature_importance_sum = 0
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            inputs = torch.tensor(X_permuted, dtype=torch.float32).unsqueeze(1)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            permuted_accuracy = (preds == labels).sum().item() / len(labels)
            importance = baseline_accuracy - permuted_accuracy
            feature_importance_sum += importance
        average_importance = feature_importance_sum / n_repeats
        importance_scores.append(average_importance)

    scaler = MinMaxScaler()
    normalized_importance = scaler.fit_transform(np.array(importance_scores).reshape(-1, 1)).flatten()

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': normalized_importance
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='#66B2FF')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    save_path = os.path.join(save_dir, 'feature_importance.png')
    plt.savefig(save_path)

    return feature_importance_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--data-dir", default=os.path.join(script_dir, "data"))
    parser.add_argument("--out-dir", default=os.path.join(script_dir, "outputs"))
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_time = time.time()
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    result_dir = os.path.join(out_dir, "CNN")
    learning_rates = args.lr
    num_epochs = args.epochs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(result_dir, f'{timestamp}_lr_{learning_rates}_epochs_{num_epochs}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    require_file(os.path.join(data_dir, "data.csv"))
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))

    feature_names = data.columns[2:-5]
    X = data.iloc[:, 2:-5].values
    y = data.iloc[:, -5].values
    coords = data.iloc[:, -4:-2].values

    X_train, X_val, X_test, y_train, y_val, y_test = custom_split_samples(X, y, coords, data, seed=args.seed)

    model, recall = train_and_evaluate(learning_rates, save_dir, num_epochs, X_train, y_train, X_val, y_val, X_test, y_test, feature_names)

    end_time = time.time()
    total_time = end_time - start_time

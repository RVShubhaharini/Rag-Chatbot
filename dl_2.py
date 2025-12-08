import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# ----------------------------
# Flexible CNN definition
# ----------------------------
class FlexibleCNN(nn.Module):
    def __init__(self, in_channels=3, 
                 filter_org="same",   # same, double, half
                 num_filters=32, 
                 kernel_size=3, 
                 activation="ReLU", 
                 use_batchnorm=False, 
                 dropout=0.0, 
                 dense_units=128, 
                 num_classes=10):
        super(FlexibleCNN, self).__init__()
        
        act_fn = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Mish": nn.Mish
        }[activation]

        filters = []
        for i in range(5):
            if filter_org == "same":
                f = num_filters
            elif filter_org == "double":
                f = num_filters * (2**i)
            elif filter_org == "half":
                f = max(4, num_filters // (2**i))  # avoid 0 filters
            filters.append(f)

        layers = []
        in_c = in_channels
        for f in filters:
            layers.append(nn.Conv2d(in_c, f, kernel_size=kernel_size, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(f))
            layers.append(act_fn())
            layers.append(nn.MaxPool2d(2, 2))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_c = f

        self.conv = nn.Sequential(*layers)

        # assuming input 224x224 -> after 5 pools: 7x7
        self.fc1 = nn.Linear(filters[-1]*7*7, dense_units)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ----------------------------
# Dataset loading
# ----------------------------
def load_datasets(data_dir, batch_size=64, val_split=0.2, augment=False):
    # Data augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # Load train dataset
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)

    # Stratified split for equal class representation
    targets = np.array([s[1] for s in full_dataset.samples])
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_split,
        stratify=targets,
        random_state=42
    )
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(datasets.ImageFolder(os.path.join(data_dir, "train"), transform=test_transform), val_idx)

    # ✅ Use val folder as test dataset
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=test_transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ----------------------------
# Training loop
# ----------------------------
def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_loader, val_loader, _ = load_datasets(
            data_dir="C:/Users/Shubha/Downloads/inaturalist_12K",  # ✅ Correct path
            batch_size=config.batch_size, 
            augment=config.data_augmentation
        )
        
        model = FlexibleCNN(
            num_filters=config.num_filters,
            activation=config.activation,
            filter_org=config.filter_org,
            use_batchnorm=config.batchnorm,
            dropout=config.dropout,
            dense_units=config.dense_units
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        for epoch in range(config.epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_acc = 100 * correct / total
            val_acc = evaluate(model, val_loader, device)

            wandb.log({"epoch": epoch+1, "train_loss": total_loss/len(train_loader), 
                       "train_acc": train_acc, "val_acc": val_acc})


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100 * correct / total


# ----------------------------
# WandB Sweep Config
# ----------------------------
sweep_config = {
    "method": "random",  # random search is efficient
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "num_filters": {"values": [32, 64]},
        "activation": {"values": ["ReLU", "GELU", "SiLU", "Mish"]},
        "filter_org": {"values": ["same", "double", "half"]},
        "data_augmentation": {"values": [True, False]},
        "batchnorm": {"values": [True, False]},
        "dropout": {"values": [0.2, 0.3]},
        "dense_units": {"values": [128, 256]},
        "batch_size": {"values": [32, 64]},
        "learning_rate": {"values": [1e-3, 5e-4]},
        "epochs": {"value": 10}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="iNaturalist-CNN")
wandb.agent(sweep_id, train_model, count=20)  # run 20 experiments

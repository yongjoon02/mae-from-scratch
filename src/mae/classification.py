"""Classification Module - CIFAR-10"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_train = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            train_size = int(0.9 * len(full_train))
            val_size = len(full_train) - train_size
            self.train_dataset, self.val_dataset = random_split(full_train, [train_size, val_size])
        
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def setup_test(self):
        """Setup test dataset explicitly"""
        self.test_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class MAEClassifier(pl.LightningModule):
    def __init__(self, encoder, num_classes=10, freeze_encoder=True, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.head = nn.Linear(768, num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        cls_token = features[:, 0]
        return self.head(cls_token)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        
        self.log('test/loss', loss)
        self.log('test/acc', acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


def visualize_classification_results(model, data_module, device, save_path):
    """Visualize classification test results"""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    model = model.to(device)
    model.eval()
    test_loader = data_module.test_dataloader()
    
    all_preds, all_labels = [], []
    sample_images, sample_preds, sample_labels = [], [], []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            if i == 0:
                sample_images = x[:16].cpu()
                sample_preds = preds[:16]
                sample_labels = y[:16].numpy()
    
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Classification Test Results', fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(3, 6, hspace=0.4, wspace=0.3)
    
    for idx in range(12):
        ax = fig.add_subplot(gs[idx // 6, idx % 6])
        img = sample_images[idx].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        ax.imshow(img)
        pred_name = class_names[sample_preds[idx]]
        true_name = class_names[sample_labels[idx]]
        color = 'green' if sample_preds[idx] == sample_labels[idx] else 'red'
        ax.set_title(f'Pred: {pred_name}\nTrue: {true_name}', fontsize=9, color=color)
        ax.axis('off')
    
    ax_cm = fig.add_subplot(gs[2, :])
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm, cbar_kws={'shrink': 0.8})
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title('Confusion Matrix')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Total samples: {len(all_labels)}")
    
    return accuracy


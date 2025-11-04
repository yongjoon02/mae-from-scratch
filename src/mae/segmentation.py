"""Segmentation Module - Oxford-IIIT Pet"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class PetSegmentationDataset(Dataset):
    def __init__(self, root, split='trainval'):
        self.dataset = OxfordIIITPet(root=root, split=split, target_types='segmentation', download=True)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor()
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        
        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_transform(mask).squeeze(0).long()
        
        mask_tensor[mask_tensor == 2] = 0
        mask_tensor[mask_tensor > 0] = 1
        
        return img_tensor, mask_tensor


def split_pet_segmentation_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test"""
    from torch.utils.data import random_split
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


class PetSegmenter(pl.LightningModule):
    def __init__(self, encoder, num_classes=2, freeze_encoder=True, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            nn.BatchNorm2d(384), nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192), nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96, num_classes, kernel_size=2, stride=2)
        )
        self.lr = lr
        self.num_classes = num_classes
        
    def forward(self, x):
        features = self.encoder(x)
        patch_features = features[:, 1:]
        B, N, C = patch_features.shape
        H = W = int(N ** 0.5)
        patch_features = patch_features.transpose(1, 2).reshape(B, C, H, W)
        seg_map = self.seg_head(patch_features)
        return seg_map
    
    def training_step(self, batch, batch_idx):
        x, mask = batch
        seg_output = self(x)
        loss = F.cross_entropy(seg_output, mask)
        pred_mask = seg_output.argmax(dim=1)
        acc = (pred_mask == mask).float().mean()
        self.log('train_seg/loss', loss, prog_bar=True)
        self.log('train_seg/acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        seg_output = self(x)
        loss = F.cross_entropy(seg_output, mask)
        pred_mask = seg_output.argmax(dim=1)
        acc = (pred_mask == mask).float().mean()
        self.log('val_seg/loss', loss, prog_bar=True)
        self.log('val_seg/acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, mask = batch
        seg_output = self(x)
        loss = F.cross_entropy(seg_output, mask)
        pred_mask = seg_output.argmax(dim=1)
        acc = (pred_mask == mask).float().mean()
        self.log('test_seg/loss', loss)
        self.log('test_seg/acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.seg_head.parameters(), lr=self.lr, weight_decay=1e-4)


def visualize_segmentation_results(model, val_dataset, device, save_path, num_samples=8):
    """Visualize segmentation test results"""
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(val_dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    
    pred_mask = None
    gt_mask = None
    
    with torch.no_grad():
        for x, gt_mask_batch in test_loader:
            x = x.to(device)
            pred_seg = model(x)
            
            x = x.cpu()
            pred_mask = torch.argmax(pred_seg, dim=1).cpu()
            gt_mask = gt_mask_batch.cpu()
            
            fig, axes = plt.subplots(3, num_samples, figsize=(24, 9))
            fig.suptitle('Pet Segmentation Test Results', fontsize=16, fontweight='bold')
            
            for idx in range(num_samples):
                img = x[idx].numpy()
                img = img * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                img = np.clip(img, 0, 1).transpose(1, 2, 0)
                
                axes[0, idx].imshow(img)
                axes[0, idx].set_title('Input', fontsize=10)
                axes[0, idx].axis('off')
                
                axes[1, idx].imshow(gt_mask[idx], cmap='gray', vmin=0, vmax=1)
                axes[1, idx].set_title('GT Mask', fontsize=10)
                axes[1, idx].axis('off')
                
                axes[2, idx].imshow(pred_mask[idx], cmap='gray', vmin=0, vmax=1)
                acc = ((pred_mask[idx] == gt_mask[idx]).float().mean() * 100).item()
                axes[2, idx].set_title(f'Pred (Acc: {acc:.1f}%)', fontsize=10)
                axes[2, idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            break
    
    if pred_mask is not None and gt_mask is not None:
        mean_iou = ((pred_mask == gt_mask).float().mean() * 100).item()
        print(f"Mean Pixel Accuracy: {mean_iou:.2f}%")
    print("Segmentation visualization completed!")


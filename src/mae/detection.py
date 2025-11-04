"""Detection Module - Oxford-IIIT Pet"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import xml.etree.ElementTree as ET


def load_bbox_from_xml(xml_path):
    """Load bounding box from XML annotation file (head ROI)"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Get bounding box (head ROI)
        obj = root.find('object')
        if obj is not None:
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                return xmin, ymin, xmax, ymax, img_width, img_height
    except Exception as e:
        print(f"Warning: Could not load bbox from {xml_path}: {e}")
    
    return None, None, None, None, None, None


class PetDetectionDataset(Dataset):
    def __init__(self, root, split='trainval', transform=None, use_xml_bbox=True):
        self.dataset_seg = OxfordIIITPet(root=root, split=split, target_types='segmentation', download=True)
        self.dataset_cat = OxfordIIITPet(root=root, split=split, target_types='category', download=False)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.num_classes = 37
        self.root = Path(root)
        self.use_xml_bbox = use_xml_bbox
        self.xml_dir = self.root / "oxford-iiit-pet" / "annotations" / "xmls"
        
        # Load image names from split file
        split_file = self.root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
        self.image_names = []
        if split_file.exists():
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Format: "Image CLASS-ID SPECIES BREED ID"
                        parts = line.split()
                        if len(parts) > 0:
                            img_name = parts[0]
                            self.image_names.append(img_name)
        
        # If we couldn't load from split file, fall back to list.txt
        if not self.image_names:
            list_file = self.root / "oxford-iiit-pet" / "annotations" / "list.txt"
            if list_file.exists():
                with open(list_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) > 0:
                                img_name = parts[0]
                                self.image_names.append(img_name)
        
        # Ensure we have the same number of image names as dataset
        if len(self.image_names) != len(self.dataset_seg):
            # Fallback: use indices as image names (will try to find XML)
            self.image_names = [f"image_{i}" for i in range(len(self.dataset_seg))]
        
    def __len__(self):
        return len(self.dataset_seg)
    
    def __getitem__(self, idx):
        img, mask = self.dataset_seg[idx]
        _, class_idx = self.dataset_cat[idx]
        
        orig_w, orig_h = img.size
        
        # Try to load bbox from XML (head ROI) if available
        if self.use_xml_bbox:
            img_name = self.image_names[idx]
            xml_path = self.xml_dir / f"{img_name}.xml"
            
            if xml_path.exists():
                xmin, ymin, xmax, ymax, img_width, img_height = load_bbox_from_xml(xml_path)
                if xmin is not None:
                    # Normalize to 224x224
                    bbox = torch.tensor([
                        xmin / img_width * 224,
                        ymin / img_height * 224,
                        xmax / img_width * 224,
                        ymax / img_height * 224
                    ], dtype=torch.float32)
                    img_tensor = self.transform(img)
                    return img_tensor, bbox, class_idx
        
        # Fallback: Generate bbox from segmentation mask (full body)
        mask_np = np.array(mask)
        non_zero = np.where(mask_np > 0)
        if len(non_zero[0]) > 0:
            y_min, y_max = non_zero[0].min(), non_zero[0].max()
            x_min, x_max = non_zero[1].min(), non_zero[1].max()
        else:
            h, w = mask_np.shape
            x_min, y_min, x_max, y_max = 0, 0, w, h
        
        bbox = torch.tensor([
            x_min / orig_w * 224,
            y_min / orig_h * 224,
            x_max / orig_w * 224,
            y_max / orig_h * 224
        ], dtype=torch.float32)
        
        img_tensor = self.transform(img)
        return img_tensor, bbox, class_idx


def split_pet_detection_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test"""
    from torch.utils.data import random_split
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


class PetDetector(pl.LightningModule):
    def __init__(self, encoder, num_classes=37, freeze_encoder=True, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.bbox_head = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 4)
        )
        self.class_head = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        self.lr = lr
        
    def forward(self, x):
        features = self.encoder(x)
        cls_token = features[:, 0]
        bbox = self.bbox_head(cls_token)
        cls = self.class_head(cls_token)
        return bbox, cls
    
    def training_step(self, batch, batch_idx):
        x, gt_boxes, gt_labels = batch
        pred_boxes, pred_cls = self(x)
        bbox_loss = F.mse_loss(pred_boxes, gt_boxes)
        cls_loss = F.cross_entropy(pred_cls, gt_labels)
        loss = bbox_loss + cls_loss
        self.log('train_det/bbox_loss', bbox_loss, prog_bar=True)
        self.log('train_det/cls_loss', cls_loss, prog_bar=True)
        self.log('train_det/total_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, gt_boxes, gt_labels = batch
        pred_boxes, pred_cls = self(x)
        bbox_loss = F.mse_loss(pred_boxes, gt_boxes)
        cls_loss = F.cross_entropy(pred_cls, gt_labels)
        loss = bbox_loss + cls_loss
        preds = torch.argmax(pred_cls, dim=1)
        acc = (preds == gt_labels).float().mean()
        self.log('val_det/bbox_loss', bbox_loss)
        self.log('val_det/cls_loss', cls_loss)
        self.log('val_det/total_loss', loss, prog_bar=True)
        self.log('val_det/acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, gt_boxes, gt_labels = batch
        pred_boxes, pred_cls = self(x)
        bbox_loss = F.mse_loss(pred_boxes, gt_boxes)
        cls_loss = F.cross_entropy(pred_cls, gt_labels)
        loss = bbox_loss + cls_loss
        preds = torch.argmax(pred_cls, dim=1)
        acc = (preds == gt_labels).float().mean()
        self.log('test_det/bbox_loss', bbox_loss)
        self.log('test_det/cls_loss', cls_loss)
        self.log('test_det/total_loss', loss)
        self.log('test_det/acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(list(self.bbox_head.parameters()) + list(self.class_head.parameters()), 
                                lr=self.lr, weight_decay=1e-4)


def visualize_detection_results(model, val_dataset, device, save_path, num_samples=8):
    """Visualize detection test results"""
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(val_dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    
    with torch.no_grad():
        for x, gt_boxes, gt_labels in test_loader:
            x = x.to(device)
            pred_boxes, pred_cls = model(x)
            
            x = x.cpu()
            pred_boxes = pred_boxes.cpu()
            pred_cls = torch.argmax(pred_cls, dim=1).cpu()
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Pet Detection Test Results', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            for idx in range(num_samples):
                ax = axes[idx]
                img = x[idx].numpy()
                img = img * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                img = np.clip(img, 0, 1).transpose(1, 2, 0)
                ax.imshow(img)
                
                gt_box = gt_boxes[idx].numpy()
                pred_box = pred_boxes[idx].numpy()
                
                rect_gt = patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                                            linewidth=2, edgecolor='green', facecolor='none', label='GT')
                rect_pred = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                                              linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Pred')
                ax.add_patch(rect_gt)
                ax.add_patch(rect_pred)
                
                gt_class = gt_labels[idx].item()
                pred_class = pred_cls[idx].item()
                color = 'green' if gt_labels[idx] == pred_cls[idx] else 'red'
                ax.set_title(f'GT: class {gt_class}\nPred: class {pred_class}', fontsize=10, color=color)
                ax.axis('off')
                if idx == 0:
                    ax.legend(loc='upper right', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            break
    
    print("Detection visualization completed!")


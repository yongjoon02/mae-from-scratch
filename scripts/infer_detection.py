"""Inference on single image - Detection"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import transforms
from src.mae import load_mae_encoder, PetDetector


def main():
    parser = argparse.ArgumentParser(description='Detect objects in single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint file (default: latest in logs/checkpoints_pet_det)')
    parser.add_argument('--save', type=str, default=None, help='Path to save visualization')
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "checkpoints").exists() else Path.cwd().parent
    WEIGHT_PATH = PROJECT_ROOT / "checkpoints" / "mae_pretrain_vit_base.pth"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # Find best checkpoint
    if args.ckpt is None:
        checkpoint_dir = LOG_DIR / "checkpoints_pet_det"
        checkpoints = list(checkpoint_dir.rglob("*.ckpt"))  # Recursive search
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}\n"
                f"Please train the model first: python scripts/train_detection.py"
            )
        
        # Parse val_det/total_loss from filename or path and find best (lowest loss)
        import re
        best_ckpt = None
        best_loss = float('inf')
        for ckpt in checkpoints:
            # Try to find val_det/total_loss in filename or path
            match = re.search(r'val_det/total_loss=([\d.]+)', str(ckpt)) or re.search(r'total_loss=([\d.]+?)(?:\.ckpt|$)', str(ckpt))
            if match:
                loss_str = match.group(1).rstrip('.')
                loss = float(loss_str)
                if loss < best_loss:
                    best_loss = loss
                    best_ckpt = ckpt
        
        if best_ckpt is None:
            # Fallback to latest if parsing fails
            best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Warning: Could not parse val_det/total_loss from filenames, using latest: {best_ckpt}")
        else:
            print(f"Using best checkpoint (val_det/total_loss={best_loss:.4f}): {best_ckpt}")
        ckpt_path = best_ckpt
    else:
        ckpt_path = Path(args.ckpt)
    
    # Load encoder
    print("\nLoading MAE Encoder...")
    encoder = load_mae_encoder(WEIGHT_PATH)
    
    # Load model
    print(f"Loading model from checkpoint: {ckpt_path}")
    
    # Validate checkpoint type by checking state_dict keys
    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    state_dict_keys = set(checkpoint.get('state_dict', {}).keys())
    
    # Check if this is a detection checkpoint
    has_detection_head = 'bbox_head' in str(state_dict_keys) or 'class_head' in str(state_dict_keys)
    has_classification_head = 'head.weight' in state_dict_keys or 'head.bias' in state_dict_keys
    has_segmentation_head = 'seg_head' in str(state_dict_keys)
    
    if has_classification_head:
        raise ValueError(
            f"Error: This is a classification checkpoint, not a detection checkpoint.\n"
            f"Please use: python scripts/infer_classification.py --image <image> --ckpt <ckpt>"
        )
    elif has_segmentation_head:
        raise ValueError(
            f"Error: This is a segmentation checkpoint, not a detection checkpoint.\n"
            f"Please use: python scripts/infer_segmentation.py --image <image> --ckpt <ckpt>"
        )
    elif not has_detection_head:
        raise ValueError(
            f"Error: Could not identify checkpoint type. "
            f"Expected detection checkpoint with 'bbox_head' or 'class_head' in state_dict."
        )
    
    model = PetDetector.load_from_checkpoint(
        str(ckpt_path),
        encoder=encoder,
        num_classes=37,
        freeze_encoder=True,
        lr=1e-3,
        strict=False
    )
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        pred_boxes, pred_cls = model(img_tensor)
        pred_boxes = pred_boxes.cpu()[0]
        pred_class = torch.argmax(pred_cls, dim=1).cpu().item()
        confidence = torch.softmax(pred_cls, dim=1)[0][pred_class].item()
    
    # Print results
    print(f"\nInput image: {img_path}")
    print(f"Predicted class: {pred_class} (confidence: {confidence*100:.2f}%)")
    print(f"Bounding box: [{pred_boxes[0]:.1f}, {pred_boxes[1]:.1f}, {pred_boxes[2]:.1f}, {pred_boxes[3]:.1f}]")
    
    # Visualization
    if args.save:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)
        
        # Draw bounding box
        rect = patches.Rectangle(
            (pred_boxes[0], pred_boxes[1]),
            pred_boxes[2] - pred_boxes[0],
            pred_boxes[3] - pred_boxes[1],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.set_title(f'Class: {pred_class} (Conf: {confidence*100:.1f}%)', fontsize=12)
        ax.axis('off')
        
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved to: {args.save}")


if __name__ == "__main__":
    main()


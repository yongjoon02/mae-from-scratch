"""Inference on single image - Segmentation"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import transforms
from src.mae import load_mae_encoder, PetSegmenter


def main():
    parser = argparse.ArgumentParser(description='Segment single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint file (default: latest in logs/checkpoints_pet_seg)')
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
        checkpoint_dir = LOG_DIR / "checkpoints_pet_seg"
        checkpoints = list(checkpoint_dir.rglob("*.ckpt"))  # Recursive search
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}\n"
                f"Please train the model first: python scripts/train_segmentation.py"
            )
        
        # Parse val_seg/loss from filename or path and find best (lowest loss)
        import re
        best_ckpt = None
        best_loss = float('inf')
        for ckpt in checkpoints:
            # Try to find val_seg/loss in filename or path
            match = re.search(r'val_seg/loss=([\d.]+)', str(ckpt)) or re.search(r'seg_loss=([\d.]+?)(?:\.ckpt|$)', str(ckpt))
            if match:
                loss_str = match.group(1).rstrip('.')
                loss = float(loss_str)
                if loss < best_loss:
                    best_loss = loss
                    best_ckpt = ckpt
        
        if best_ckpt is None:
            # Fallback to latest if parsing fails
            best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Warning: Could not parse val_seg/loss from filenames, using latest: {best_ckpt}")
        else:
            print(f"Using best checkpoint (val_seg/loss={best_loss:.4f}): {best_ckpt}")
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
    
    # Check if this is a segmentation checkpoint
    has_segmentation_head = 'seg_head' in str(state_dict_keys)
    has_classification_head = 'head.weight' in state_dict_keys or 'head.bias' in state_dict_keys
    has_detection_head = 'bbox_head' in str(state_dict_keys) or 'class_head' in str(state_dict_keys)
    
    if has_classification_head:
        raise ValueError(
            f"Error: This is a classification checkpoint, not a segmentation checkpoint.\n"
            f"Please use: python scripts/infer_classification.py --image <image> --ckpt <ckpt>"
        )
    elif has_detection_head:
        raise ValueError(
            f"Error: This is a detection checkpoint, not a segmentation checkpoint.\n"
            f"Please use: python scripts/infer_detection.py --image <image> --ckpt <ckpt>"
        )
    elif not has_segmentation_head:
        raise ValueError(
            f"Error: Could not identify checkpoint type. "
            f"Expected segmentation checkpoint with 'seg_head' in state_dict."
        )
    
    model = PetSegmenter.load_from_checkpoint(
        str(ckpt_path),
        encoder=encoder,
        num_classes=2,
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
        pred_seg = model(img_tensor)
        pred_mask = torch.argmax(pred_seg, dim=1).cpu()[0]
    
    # Print results
    foreground_ratio = (pred_mask == 1).float().mean().item()
    print(f"\nInput image: {img_path}")
    print(f"Foreground ratio: {foreground_ratio*100:.2f}%")
    
    # Visualization
    if args.save:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        # Overlay
        img_np = np.array(img)
        overlay = img_np.copy()
        mask_np = pred_mask.numpy().astype(np.uint8)
        overlay[mask_np == 1] = overlay[mask_np == 1] * 0.7 + np.array([255, 0, 0]) * 0.3
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved to: {args.save}")


if __name__ == "__main__":
    main()


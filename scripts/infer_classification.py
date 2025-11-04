"""Inference on single image - Classification"""

import sys
import argparse
from pathlib import Path
from PIL import Image

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import transforms
from src.mae import load_mae_encoder, MAEClassifier


def main():
    parser = argparse.ArgumentParser(description='Classify single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint file (default: latest in logs/checkpoints)')
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "checkpoints").exists() else Path.cwd().parent
    WEIGHT_PATH = PROJECT_ROOT / "checkpoints" / "mae_pretrain_vit_base.pth"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # Find best checkpoint
    if args.ckpt is None:
        checkpoint_dir = LOG_DIR / "checkpoints"
        checkpoints = list(checkpoint_dir.rglob("*.ckpt"))  # Recursive search
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}\n"
                f"Please train the model first: python scripts/train_classification.py"
            )
        
        # Parse val_acc from filename or path and find best
        import re
        best_ckpt = None
        best_acc = -1
        for ckpt in checkpoints:
            # Try to find val_acc in filename or path
            match = re.search(r'val_acc=([\d.]+)', str(ckpt)) or re.search(r'acc=([\d.]+?)(?:\.ckpt|$)', str(ckpt))
            if match:
                acc_str = match.group(1).rstrip('.')
                acc = float(acc_str)
                if acc > best_acc:
                    best_acc = acc
                    best_ckpt = ckpt
        
        if best_ckpt is None:
            # Fallback to latest if parsing fails
            best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Warning: Could not parse val_acc from filenames, using latest: {best_ckpt}")
        else:
            print(f"Using best checkpoint (val_acc={best_acc:.4f}): {best_ckpt}")
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
    
    # Check if this is a classification checkpoint
    has_classification_head = 'head.weight' in state_dict_keys or 'head.bias' in state_dict_keys
    has_segmentation_head = 'seg_head' in str(state_dict_keys)
    has_detection_head = 'bbox_head' in str(state_dict_keys) or 'class_head' in str(state_dict_keys)
    
    if has_segmentation_head:
        raise ValueError(
            f"Error: This is a segmentation checkpoint, not a classification checkpoint.\n"
            f"Please use: python scripts/infer_segmentation.py --image <image> --ckpt <ckpt>"
        )
    elif has_detection_head:
        raise ValueError(
            f"Error: This is a detection checkpoint, not a classification checkpoint.\n"
            f"Please use: python scripts/infer_detection.py --image <image> --ckpt <ckpt>"
        )
    elif not has_classification_head:
        raise ValueError(
            f"Error: Could not identify checkpoint type. "
            f"Expected classification checkpoint with 'head.weight' in state_dict."
        )
    
    model = MAEClassifier.load_from_checkpoint(
        str(ckpt_path),
        encoder=encoder,
        num_classes=10,
        freeze_encoder=True,
        lr=1e-3,
        weight_decay=1e-4,
        strict=False
    )
    model = model.to(device)
    model.eval()
    
    # Class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load and preprocess image
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    # Print results
    print(f"\nInput image: {img_path}")
    print(f"Predicted class: {class_names[pred_class]} (confidence: {confidence*100:.2f}%)")
    print("\nTop 3 predictions:")
    top3_probs, top3_indices = torch.topk(probs[0], 3)
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
        print(f"  {i+1}. {class_names[idx.item()]}: {prob.item()*100:.2f}%")


if __name__ == "__main__":
    main()


"""Evaluate Detection Task - Load checkpoint and evaluate"""

import sys
import argparse
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
from src.mae import (
    load_mae_encoder,
    PetDetectionDataset,
    PetDetector,
    visualize_detection_results,
    split_pet_detection_dataset
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Detection Model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint file (default: latest in logs/checkpoints_pet_det)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load encoder
    print("\nLoading MAE Encoder...")
    encoder = load_mae_encoder(WEIGHT_PATH)
    print("MAE Encoder loaded")
    
    # Dataset
    print("\nLoading Pet Detection Dataset...")
    pet_det_dataset = PetDetectionDataset(root=str(PROJECT_ROOT / "data"), split='trainval')
    pet_det_train, pet_det_val, pet_det_test = split_pet_detection_dataset(
        pet_det_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    pet_det_test_loader = DataLoader(pet_det_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(pet_det_test)}")
    
    # Load model from checkpoint
    print(f"\nLoading model from checkpoint: {ckpt_path}")
    model = PetDetector.load_from_checkpoint(
        str(ckpt_path),
        encoder=encoder,
        num_classes=37,
        freeze_encoder=True,
        lr=1e-3
    )
    print("Model loaded")
    
    # Trainer for evaluation
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=True
    )
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_results = trainer.test(model, pet_det_test_loader)
    print("Test evaluation completed!")
    
    # Visualization
    print("\nGenerating visualization...")
    visualize_detection_results(
        model=model,
        val_dataset=pet_det_test,
        device=device,
        save_path=LOG_DIR / "pet_detection_test_results.png"
    )
    
    print(f"\nResults saved to: {LOG_DIR / 'pet_detection_test_results.png'}")


if __name__ == "__main__":
    main()


"""Evaluate Classification Task - Load checkpoint and evaluate on test set"""

import sys
import argparse
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pytorch_lightning import Trainer
from src.mae import (
    load_mae_encoder,
    CIFAR10DataModule,
    MAEClassifier,
    visualize_classification_results
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Classification Model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint file (default: latest in logs/checkpoints)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
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
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load encoder
    print("\nLoading MAE Encoder...")
    encoder = load_mae_encoder(WEIGHT_PATH)
    print("MAE Encoder loaded")
    
    # Data module
    data_module = CIFAR10DataModule(
        data_dir=str(PROJECT_ROOT / "data"),
        batch_size=args.batch_size,
        num_workers=0
    )
    data_module.setup('test')
    
    # Load model from checkpoint
    print(f"\nLoading model from checkpoint: {ckpt_path}")
    model = MAEClassifier.load_from_checkpoint(
        str(ckpt_path),
        encoder=encoder,
        num_classes=10,
        freeze_encoder=True,
        lr=1e-3,
        weight_decay=1e-4
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
    test_results = trainer.test(model, data_module)
    print("Test evaluation completed!")
    
    # Visualization
    print("\nGenerating visualization...")
    accuracy = visualize_classification_results(
        model=model,
        data_module=data_module,
        device=device,
        save_path=LOG_DIR / "classification_test_results.png"
    )
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {LOG_DIR / 'classification_test_results.png'}")


if __name__ == "__main__":
    main()


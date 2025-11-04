"""Train Detection Task - Oxford-IIIT Pet"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.mae import (
    load_mae_encoder,
    PetDetectionDataset,
    PetDetector,
    visualize_detection_results,
    split_pet_detection_dataset
)

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "checkpoints").exists() else Path.cwd().parent
    WEIGHT_PATH = PROJECT_ROOT / "checkpoints" / "mae_pretrain_vit_base.pth"
    LOG_DIR = PROJECT_ROOT / "logs"
    
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
    
    pet_det_train_loader = DataLoader(pet_det_train, batch_size=32, shuffle=True, num_workers=0)
    pet_det_val_loader = DataLoader(pet_det_val, batch_size=32, shuffle=False, num_workers=0)
    pet_det_test_loader = DataLoader(pet_det_test, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(pet_det_train)}, Val samples: {len(pet_det_val)}, Test samples: {len(pet_det_test)}")
    
    # Model
    pet_detector = PetDetector(
        encoder=encoder,
        num_classes=37,
        freeze_encoder=True,
        lr=1e-3
    )
    
    # Loggers
    tensorboard_logger = TensorBoardLogger(LOG_DIR, name="pet_detection")
    csv_logger = CSVLogger(LOG_DIR, name="pet_detection")
    
    # Trainer
    from pytorch_lightning import Trainer
    pet_det_trainer = Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        logger=[tensorboard_logger, csv_logger],
        callbacks=[
            ModelCheckpoint(
                dirpath=LOG_DIR / "checkpoints_pet_det",
                filename="pet-detector-{epoch:02d}-{val_det/total_loss:.4f}",
                monitor="val_det/total_loss",
                mode="min",
                save_top_k=3
            ),
            EarlyStopping(
                monitor="val_det/total_loss",
                patience=20,
                mode="min",
                verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ],
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # Training
    print("\nStarting Pet Detection training...")
    pet_det_trainer.fit(pet_detector, pet_det_train_loader, pet_det_val_loader)
    print("Pet Detection training completed!")
    
    # Validation
    print("\nEvaluating on validation set...")
    pet_det_trainer.validate(pet_detector, pet_det_val_loader)
    print("Pet Detection validation completed!")
    
    # Testing
    print("\nEvaluating on test set...")
    pet_det_trainer.test(pet_detector, pet_det_test_loader)
    print("Pet Detection test completed!")
    
    # Visualization
    print("\nGenerating visualization...")
    visualize_detection_results(
        model=pet_detector,
        val_dataset=pet_det_test,
        device=device,
        save_path=LOG_DIR / "pet_detection_test_results.png"
    )
    
    print(f"\nLogs saved to: {LOG_DIR}")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR}")

if __name__ == "__main__":
    main()


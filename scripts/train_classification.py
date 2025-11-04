"""Train Classification Task - CIFAR-10"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.mae import (
    load_mae_encoder,
    CIFAR10DataModule,
    MAEClassifier,
    visualize_classification_results
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
    
    # Data module
    data_module = CIFAR10DataModule(
        data_dir=str(PROJECT_ROOT / "data"),
        batch_size=64,
        num_workers=0
    )
    
    # Model
    model = MAEClassifier(
        encoder=encoder,
        num_classes=10,
        freeze_encoder=True,
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # Loggers and callbacks
    tensorboard_logger = TensorBoardLogger(LOG_DIR, name="mae_classification")
    csv_logger = CSVLogger(LOG_DIR, name="mae_classification")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=LOG_DIR / "checkpoints",
        filename="mae-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=20,
        mode="min",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    from pytorch_lightning import Trainer
    trainer = Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        logger=[tensorboard_logger, csv_logger],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # Training
    print("\nStarting Classification training...")
    trainer.fit(model, data_module)
    print("Training completed!")
    
    # Validation
    print("\nEvaluating on validation set...")
    trainer.validate(model, data_module)
    print("Validation completed!")
    
    # Testing
    print("\nEvaluating on test set...")
    trainer.test(model, data_module)
    print("Test completed!")
    
    # Visualization
    print("\nGenerating visualization...")
    visualize_classification_results(
        model=model,
        data_module=data_module,
        device=device,
        save_path=LOG_DIR / "classification_test_results.png"
    )
    
    print(f"\nLogs saved to: {LOG_DIR}")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR}")

if __name__ == "__main__":
    main()


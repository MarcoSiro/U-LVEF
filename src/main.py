
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping           

from utilities import LightningModel, CamusDataModule, UNet, plot_loss_curves, plot_segmentation_results, UNetMini

def main():
    input_channels = 1   #Grey scale ECO signals (1 channel)
    num_classes = 4      #4 macro-categories of the Camus dataset: Background (0), Myocardium (1), Left Ventricle (2), Left Atrium (3)

    pytorch_model = UNetMini(in_channels=input_channels, out_channels=num_classes) #Inizializing Pytorch model

    L.pytorch.seed_everything(123)

    dm = CamusDataModule(data_path="./data/camus_2D", batch_size=16, num_workers=4) #Inzializing DataModule

    lightning_model = LightningModel(
        model=pytorch_model, 
        learning_rate=1e-4, 
        num_classes=num_classes
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="logs/checkpoints/",
        filename="best-camus-unet-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        mode="min",
    )       #Model checkpoint callback to save the best model based on validation loss during training

    '''early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=10, 
        verbose=True, 
        mode="min"
    )'''       #Early stopping callback to prevent overfitting and save training time

    #Configuration of Lightning Trainer
    trainer = L.Trainer(
        max_epochs=100,          # We have set a high number of epochs, but the training will stop early if the validation loss does not improve for 10 consecutive epochs (due to EarlyStopping callback)
        accelerator="auto", 
        devices=1,
        logger=CSVLogger(save_dir="logs/", name="camus-unet"), # Nome aggiornato
        deterministic=True,
        callbacks=[checkpoint_callback]     #Consider adding early_stop_callback (not suggested with cosine annealing + warm restarts)
    )

    print("\n\nStarting Training...")       
    trainer.fit(model=lightning_model, datamodule=dm)

    print("\n\nStarting Final Test...")     
    trainer.test(model=lightning_model, datamodule=dm)

    log_dir = trainer.logger.log_dir
    print(f"\n\nSaving graphics in the folder: {log_dir}")

    class_names = ['Background', 'Myocardium', 'Left Ventricle', 'Left Atrium']

    #Loss plot (Train vs Val)
    plot_loss_curves(log_dir) 
    
    # Test Dice Score (Average)
    print("\n\n" + "="*40)
    print("FINAL SEGMENTATION REPORT ")
    print("="*40)
    print(f"Test Dice Score (Macro): {lightning_model.final_dice_saved:.4f}")
    print("="*40)

    # Segmentation results visualization
    plot_segmentation_results(lightning_model.model, dm, class_names, log_dir, num_examples=3)   

if __name__ == '__main__':
    main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
from torch.utils.data import Dataset, DataLoader
import glob
import albumentations as A
import cv2
import matplotlib.colors as mcolors



class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2 times """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(   #kernel_size=3, padding=1  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        
        # ---------------- ENCODER ----------------
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # ---------------- BOTTLENECK ----------------
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # ---------------- DECODER ----------------
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        # ---------------- FINAL LAYER ----------------
        # Final 1x1 convolution to get the desired number of output channels (classes)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # variables for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) 
        
        # concatenation with skip connections and upsampling
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1) # Skip connection 1
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1) # Skip connection 2
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1) # Skip connection 3
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1) # Skip connection 4
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits

class UNetMini(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        
        # ---------------- ENCODER ----------------
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # ---------------- BOTTLENECK ----------------
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # ---------------- DECODER ----------------
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(32, 16)
        
        # ---------------- FINAL LAYER ----------------
        # Final 1x1 convolution to get the desired number of output channels (classes)
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # variables for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) 
        
        # concatenation with skip connections and upsampling
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)   # Skip connection 1
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)   # Skip connection 2
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)   # Skip connection 3
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)   # Skip connection 4
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits


class CamusDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Loading .npy files
        img_frame = np.load(self.image_paths[idx])
        mask_frame = np.load(self.mask_paths[idx])
            
        # Normalizing
        img_frame = img_frame.astype(np.float32)
        img_min, img_max = img_frame.min(), img_frame.max()
        if img_max > img_min:
            img_frame = (img_frame - img_min) / (img_max - img_min)
            
        mask_frame = mask_frame.astype(np.int32) 
        
        # Data Augmentation (Albumentations)
        if self.transform is not None:
            augmented = self.transform(image=img_frame, mask=mask_frame)
            img_frame = augmented['image']
            mask_frame = augmented['mask']
            
        # Conversion in Tensors PyTorch
        img_tensor = torch.tensor(img_frame, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_frame, dtype=torch.long)
        
        return img_tensor, mask_tensor

class CamusDataModule(L.LightningDataModule):
    def __init__(self, data_path="./data/camus_2D", batch_size=16, num_workers=0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        search_path = os.path.join(self.data_path, "Images", "*.npy")
        valid_image_paths = sorted(glob.glob(search_path))
        
        if len(valid_image_paths) == 0:
            raise FileNotFoundError(f"No file .npy found in {search_path}.")

        patient_ids = []
        for path in valid_image_paths:
            basename = os.path.basename(path)
            pat_id = basename.split('_')[0] 
            if pat_id not in patient_ids:
                patient_ids.append(pat_id)

        total_patients = len(patient_ids)
        train_split = int(0.8 * total_patients)
        val_split = int(0.9 * total_patients)

        train_patients = set(patient_ids[:train_split])
        val_patients = set(patient_ids[train_split:val_split])
        test_patients = set(patient_ids[val_split:])

        self.train_imgs, self.train_masks = [], []
        self.val_imgs, self.val_masks = [], []
        self.test_imgs, self.test_masks = [], []

        for img_path in valid_image_paths:
            mask_path = img_path.replace("Images", "Masks")
            pat_id = os.path.basename(img_path).split('_')[0]

            if pat_id in train_patients:
                self.train_imgs.append(img_path)
                self.train_masks.append(mask_path)
            elif pat_id in val_patients:
                self.val_imgs.append(img_path)
                self.val_masks.append(mask_path)
            else:
                self.test_imgs.append(img_path)
                self.test_masks.append(mask_path)

        # Data Augmentation
        self.train_transform = A.Compose([
            A.Resize(256, 256),
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
                     scale=(0.95, 1.05), rotate=(-10, 10), p=0.5, mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ])
        
        self.val_transform = A.Compose([A.Resize(256, 256)])

        self.train_dataset = CamusDataset(self.train_imgs, self.train_masks, transform=self.train_transform)
        self.val_dataset = CamusDataset(self.val_imgs, self.val_masks, transform=self.val_transform)
        self.test_dataset = CamusDataset(self.test_imgs, self.test_masks, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, num_classes=4):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.num_classes = num_classes

        self.save_hyperparameters(ignore=["model"])

        # Calculation of Dice score for multi-class segmentation, with average='macro' to compute the score for each class and then take the average, which is a common approach for multi-class segmentation tasks. This allows us to evaluate the performance of the model across all classes, including the background, myocardium, left ventricle, and left atrium in the Camus dataset.
        self.train_dice = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_dice = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_dice = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.final_dice_saved = None

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_masks = batch
        
        # Logits output: shape [Batch, 4, Altezza, Larghezza]
        logits = self(features) 

        # Loss calculation: CrossEntropy (contains softmax)
        loss = F.cross_entropy(logits, true_masks)
        
        preds = torch.argmax(logits, dim=1) 
        
        return loss, true_masks, preds

    def training_step(self, batch, batch_idx):
        loss, true_masks, preds = self._shared_step(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_masks, preds = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_dice(preds, true_masks)
        self.log("val_dice", self.val_dice, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_masks, preds = self._shared_step(batch)
        
        self.test_dice(preds, true_masks)
        self.log("test_dice", self.test_dice)

    def configure_optimizers(self):
        # AdamW with starting learning rate 
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        # Gold Standard: Cosine Annealing Warm Restarts
        # T_0 = 10: First restart after 10 epochs
        # T_mult = 2: The subsequent cycles double in length (epoch 10 -> 30 -> 70).
        # This allows the network to explore a lot at the beginning and do "fine-tuning" extended at the end.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
            },
        }
    
    def on_test_epoch_end(self): 
        self.final_dice_saved = self.test_dice.compute().item()
    

    
def plot_loss_curves(log_dir):
    """Generates and saves Loss and Dice Score curves from Lightning logs."""
    metrics_path = os.path.join(log_dir, "metrics.csv")
    
    if not os.path.exists(metrics_path):
        print(f"File metrics.csv not found in {log_dir}")
        return
        
    df = pd.read_csv(metrics_path)
    
    if 'epoch' in df.columns:
        # Group metrics by epoch to align training and validation steps
        df_grouped = df.groupby('epoch').mean()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # --- PLOT 1: ERROR CURVE (CROSS ENTROPY LOSS) ---
        if 'train_loss' in df_grouped.columns and 'val_loss' in df_grouped.columns:
            axes[0].plot(df_grouped.index, df_grouped['train_loss'], label='Train Loss', marker='o', linewidth=2)
            axes[0].plot(df_grouped.index, df_grouped['val_loss'], label='Validation Loss', marker='s', linewidth=2)
            axes[0].set_title('Error Trend (Cross Entropy Loss)', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].legend(fontsize=12)
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
        # --- PLOT 2: PRECISION CURVE (DICE SCORE) ---
        # Modified: Removed train_dice to allow plotting of val_dice only
        if 'val_dice' in df_grouped.columns:
            axes[1].plot(df_grouped.index, df_grouped['val_dice'], label='Validation Dice', marker='s', linewidth=2, color='#ff7f0e')
            axes[1].set_title('Precision Trend (Dice Score)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Dice Score', fontsize=12)
            axes[1].legend(fontsize=12)
            axes[1].grid(True, linestyle='--', alpha=0.7)
            axes[1].set_ylim([0.0, 1.05]) # Dice score is mathematically constrained between 0 and 1
            
        plt.tight_layout()
        filename = 'training_curves.png'
        filepath = os.path.join(log_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"-> Saved: {filename}")
        
        plt.close()


def plot_segmentation_results(model, datamodule, class_names, log_dir, num_examples=3):
    print("\n--- Generation of some segmentation results ---")
    
    # Model in evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Single batch from the test set
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    images, true_masks = batch
    images = images.to(device)
    
    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1) 
        
    # From tensors to numpy arrays for plotting
    images_np = images.cpu().numpy()
    true_masks_np = true_masks.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    # Creation of a custom colormap for the segmentation masks, where each class is represented by a specific color:
    # 0: background (Black/Transparent), 1: Myocardium (Green), 2: Left Ventricle (Red), 3: Left Atrium (Blue)
    colors = ['black', '#2ca02c', '#d62728', '#1f77b4'] 
    cmap = mcolors.ListedColormap(colors)
    
    num_to_plot = min(num_examples, images_np.shape[0])
    
    # Generating plots
    for i in range(num_to_plot):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Estraiamo l'immagine 2D rimuovendo la dimensione del canale (1, 256, 256) -> (256, 256)
        img_2d = images_np[i, 0] 
        gt = true_masks_np[i]
        pred = preds_np[i]
        
        # Plot A: Original Image (Grayscale)
        axes[0].imshow(img_2d, cmap='gray')
        axes[0].set_title("Original Image", fontweight='bold')
        axes[0].axis('off')
        
        # Plot B: Ground Truth
        axes[1].imshow(img_2d, cmap='gray')
        # np.ma.masked_where to make the background (class 0) transparent, so that only the segmented classes are colored according to the colormap
        gt_masked = np.ma.masked_where(gt == 0, gt)
        axes[1].imshow(gt_masked, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
        axes[1].set_title("Ground Truth", fontweight='bold')
        axes[1].axis('off')
        
        # Plot C: U-Net Prediction
        axes[2].imshow(img_2d, cmap='gray')
        pred_masked = np.ma.masked_where(pred == 0, pred)
        axes[2].imshow(pred_masked, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
        axes[2].set_title("U-Net Prediction", fontweight='bold')
        axes[2].axis('off')
        
        # Legend
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=colors[j], label=class_names[j]) for j in range(1, 4)]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=12)
        
        plt.tight_layout()
        
        # Saving
        filename = f"segmentation_test_case_{i+1}.png"
        filepath = os.path.join(log_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"-> Salvato: {filename}")
        
        plt.close()

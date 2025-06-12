import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import gc

class RunwayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size

        # Get image files
        self.image_files = []
        all_files = os.listdir(image_dir)
        for file in all_files:
            if file.lower().endswith('.png') and not file.endswith('.mask.png'):
                base_name = os.path.splitext(file)[0]
                mask_file = base_name + '.mask.png'
                mask_path = os.path.join(mask_dir, mask_file)

                if os.path.exists(mask_path):
                    self.image_files.append(file)

        print(f"Found {len(self.image_files)} valid image-mask pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_file = os.path.splitext(img_file)[0] + '.mask.png'
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Load and resize images to reduce memory usage
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size)

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
    
# U NET MİMARİSİ 
class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(LightweightUNet, self).__init__()

        # Reduced channel sizes for memory efficiency
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        # Smaller bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)

        # Final layer
        self.final = nn.Conv2d(32, out_channels, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # Decoder
        d3 = self.dec3(torch.cat([F.interpolate(b, e3.shape[2:], mode='bilinear', align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), e1], 1))

        return self.final(d1)

class UNetTrainer:
    def __init__(self, model_path=None, device=None, input_size=(256, 256)):
        """U-Net Training System"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

        # Initialize model
        self.model = LightweightUNet().to(self.device)
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Pre-trained model loaded from {model_path}")

        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)

        print(f"UNetTrainer initialized on {self.device}")
        print(f"Input size: {input_size}")

    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def create_dataloader(self, image_dir, mask_dir, batch_size=2, shuffle=True):
        """Create data loader for training"""
        dataset = RunwayDataset(image_dir, mask_dir, target_size=self.input_size)
        
        # Only use pin_memory if CUDA is available
        use_pin_memory = torch.cuda.is_available()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0, 
            pin_memory=use_pin_memory
        )
        return dataloader, len(dataset)

    def train_model(self, image_dir, mask_dir, epochs=10, batch_size=2, learning_rate=0.001, save_dir="unet_weights"):
        """Memory-optimized training"""
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loader
        train_loader, dataset_size = self.create_dataloader(image_dir, mask_dir, batch_size, shuffle=True)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Mixed precision training only if CUDA is available
        use_amp = torch.cuda.is_available()
        scaler = None
        if use_amp:
            try:
                scaler = torch.amp.GradScaler('cuda')
            except:
                # Fallback for older PyTorch versions
                scaler = torch.cuda.amp.GradScaler()
        
        print(f"Starting training on {self.device}")
        print(f"Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
        print(f"Dataset size: {dataset_size}, Batch size: {batch_size}")
        print(f"Total batches per epoch: {len(train_loader)}")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0

            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(self.device, non_blocking=torch.cuda.is_available())
                masks = masks.to(self.device, non_blocking=torch.cuda.is_available())

                optimizer.zero_grad()

                # Forward pass with optional mixed precision
                if use_amp and scaler:
                    try:
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(images)
                            loss = criterion(outputs, masks)
                    except:
                        # Fallback for older PyTorch versions
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = criterion(outputs, masks)
                    
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training without mixed precision
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Clear intermediate results
                del images, masks, outputs

                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
                    self.clear_memory()

            scheduler.step()
            avg_loss = total_loss / batch_count
            print(f'Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}')

            # Save model periodically
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                save_path = os.path.join(save_dir, f'lightweight_unet_epoch_{epoch+1}.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f'Model checkpoint saved: {save_path}')

            self.clear_memory()

        # Save final model
        final_model_path = os.path.join(save_dir, 'lightweight_unet_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        print(f'Final model saved: {final_model_path}')

        return self.model

    def validate_model(self, image_dir, mask_dir, batch_size=1):
        """Validate the trained model"""
        val_loader, dataset_size = self.create_dataloader(image_dir, mask_dir, batch_size, shuffle=False)
        
        self.model.eval()
        total_loss = 0
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"Starting validation on {dataset_size} samples...")
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(self.device, non_blocking=False)  # CPU için non_blocking=False
                masks = masks.to(self.device, non_blocking=False)
                
                # Forward pass (no mixed precision needed for validation)
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Clear memory
                del images, masks, outputs
                
                if batch_idx % 20 == 0:
                    print(f'Validation batch {batch_idx+1}/{len(val_loader)}')
                    self.clear_memory()
        
        avg_val_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        return avg_val_loss

    def save_model(self, save_path):
        """Save model weights"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, model_path):
        """Load model weights"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from: {model_path}")
        else:
            print(f"Model file not found: {model_path}")

def main():
    """Main training function"""
    # Training configuration
    image_dir = r"split_dataset\PNG"  
    mask_dir = r"split_dataset\MASK_PNG"    
    save_dir = r"trained_models\unet_weights"
    
    # Training parameters
    config = {
        'epochs': 10,
        'batch_size': 4,
        'learning_rate': 0.0001,
        'input_size': (256, 256)
    }

    print("🚀 Initializing U-Net Training System...")
    trainer = UNetTrainer(input_size=config['input_size'])

    print("🎯 Starting training...")
    trained_model = trainer.train_model(
        image_dir=image_dir,
        mask_dir=mask_dir,
        epochs=config['epochs'],         
        batch_size=config['batch_size'],      
        learning_rate=config['learning_rate'],
        save_dir=save_dir
    )

    print("✅ Training completed!")
    print(f"📁 Model checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()
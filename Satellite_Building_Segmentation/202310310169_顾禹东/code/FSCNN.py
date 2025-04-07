import os
import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF

# ------------------ Data Augmentation ------------------
class Augment:
    def __call__(self, image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            angle = random.randint(-10, 10)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        return image, mask

# ------------------ Dataset ------------------
class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        if self.augment:
            image, mask = self.augment(image, mask)

        mask = (mask > 0.5).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask

# ------------------ Fast-SCNN ------------------
class FastSCNN(nn.Module):
    def __init__(self, n_classes):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.global_feature_extractor = nn.Sequential(
            nn.Conv2d(48, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(128, n_classes, 1)
        )

    def forward(self, x):
        size = x.size()[2:]
        x = self.learning_to_downsample(x)
        x = self.global_feature_extractor(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x

# ------------------ Dice Loss ------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# ------------------ Training ------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = "./dataset/images"
    mask_dir = "./dataset/masks"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    augment = Augment()
    dataset = SatelliteDataset(image_dir, mask_dir, transform=transform, augment=augment)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = FastSCNN(n_classes=1).to(device)
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_iou = 0.0
    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        total_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5

                intersection = (preds & masks.bool()).float().sum((1, 2, 3))
                union = (preds | masks.bool()).float().sum((1, 2, 3))
                iou = (intersection / (union + 1e-6)).mean().item()
                total_iou += iou

        avg_iou = total_iou / len(val_loader)
        print(f"Validation IoU: {avg_iou:.4f}")

        scheduler.step(avg_iou)

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

if __name__ == '__main__':
    train()

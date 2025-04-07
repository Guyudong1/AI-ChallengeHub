import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn


# ------------------ Dataset ------------------
class SatelliteDatasetTest:
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.images[idx]


# ------------------ Fast-SCNN ------------------
class FastSCNN(nn.Module):
    def __init__(self, n_classes):
        super(FastSCNN, self).__init__()
        # Define each block according to Fast-SCNN paper
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


# ------------------ Prediction Function ------------------
def predict(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Convert to binary mask

            # Save the predicted masks
            for i in range(images.size(0)):
                pred = preds[i].cpu().numpy().squeeze()  # Convert to numpy array
                pred_img = (pred * 255).astype(np.uint8)  # Convert to 0-255 range
                pred_img = Image.fromarray(pred_img)  # Convert to image
                pred_img.save(os.path.join('predictions', filenames[i]))  # Save to output folder


# ------------------ Main ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths
    test_image_dir = "dataset/tests"
    output_mask_dir = "mini_dataset/output_masks"

    # Ensure output directory exists
    os.makedirs(output_mask_dir, exist_ok=True)

    # Define transformations for the test images
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Load test dataset
    test_dataset = SatelliteDatasetTest(image_dir=test_image_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Load the trained model
    model = FastSCNN(n_classes=1).to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    # Predict on test dataset
    predict(model, test_loader, device)


if __name__ == '__main__':
    main()

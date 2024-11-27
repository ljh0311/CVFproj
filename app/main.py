import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
import os

# Configuration
class Config:
    # Hyperparameters
    EPOCHS = 10
    LR = 1e-3
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    DATA_DIR = "data/plantDataset/train"
    TEST_DIR = "data/test"
    VAL_DIR = "data/plantDataset/valid"
    
    # Model save path
    MODEL_SAVE_PATH = "models/best_model.pth"
    
    # Normalization parameters
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

# Data transformations
class DataTransforms:
    @staticmethod
    def get_transforms():
        train_transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])

        return train_transform, val_transform

# Dataset
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []

        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                self.images.append(
                    (os.path.join(class_path, img_name), self.class_to_idx[class_name])
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Model
def create_model(num_classes):
    # Load ResNet50 with pre-trained weights
    model = models.resnet50(pretrained=True)

    # Freeze early layers to prevent overfitting
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)

# Function to Train One Epoch
def train_epoch(model, train_loader, criterion, device, optimizer):
    model.train()
    epoch_loss = 0
    num_batches = len(train_loader)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / num_batches

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for evaluation.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: Validation accuracy and list of average losses over batches.
    """
    model.eval()  # Set model to evaluation mode
    datasize = 0  # Total number of samples evaluated
    accuracy = 0  # Total accuracy over batches
    total_loss = 0  # Cumulative loss over batches
    losses = []  # Store losses for each batch

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Predicted class indices

            if criterion is not None:
                loss = criterion(outputs, labels)  # Compute batch loss
                total_loss += loss.item()
                losses.append(loss.item())

            # Update accuracy
            accuracy += torch.sum(preds == labels).item()
            datasize += labels.size(0)

    # Compute overall accuracy and average loss
    accuracy /= datasize
    avg_loss = total_loss / len(dataloader) if criterion else None

    return accuracy, losses

# Create dataset and dataloaders
train_dataset = CustomImageDataset(Config.DATA_DIR, transform=DataTransforms.get_transforms()[0])
val_dataset = CustomImageDataset(Config.VAL_DIR, transform=DataTransforms.get_transforms()[1])
test_dataset = CustomImageDataset(Config.TEST_DIR, transform=DataTransforms.get_transforms()[1])
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0),
    "val": DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0),
    "test": DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0),
}

# Create the model
num_classes = len(train_dataset.classes)
model = create_model(num_classes)
model = model.to(Config.DEVICE)

# Create the model
num_classes = len(train_dataset.classes)
model = create_model(num_classes)
model = model.to(Config.DEVICE)

# device=torch.device("cuda:0")
device = torch.device("cpu")

best_valid_loss = float("inf")

for epoch in range(Config.EPOCHS):

    start_time = time.monotonic()

    # Train the model for one epoch
    train_loss = train_epoch(model, dataloaders["train"], criterion, Config.DEVICE, optimizer)

    valid_accuracy, valid_losses = evaluate(
        model, dataloaders["val"], criterion, Config.DEVICE
    )
    valid_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None

    # Save the model if validation loss improves
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    end_time = time.monotonic()

    # Calculate epoch duration
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    # Print epoch statistics
    print(f"Epoch: {epoch + 1:02} | Epoch Time: {int(epoch_mins)}m {int(epoch_secs)}s")
    print(f"\tTrain Loss: {train_loss:.3f}")
    print(
        f"\tValid Loss: {valid_loss:.3f} | Valid Accuracy: {valid_accuracy * 100:.2f}%"
    )

    # After validation in your epoch loop
    scheduler.step(valid_loss)

# Save the Model
torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
print("Model saved!")

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

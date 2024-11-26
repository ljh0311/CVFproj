import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models

# Hyperparameters
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define train, val, and test splits
trainval_split = 50000  # Number of samples for training/validation split
val_start = trainval_split
val_end = 60000

# parameters
batchsize = 32
maxnumepochs = 3
lrates = [0.01, 0.001]

# Paths and Dataset
DATA_DIR = "C:/Users/user/Documents/SITstuffs/CompVis/CVFproj/data/plantDataset/train"
TEST_DIR = "C:/Users/user/Documents/SITstuffs/CompVis/CVFproj/data/test"
VAL_DIR = "C:/Users/user/Documents/SITstuffs/CompVis/CVFproj/data/plantDataset/valid"
IMAGE_SIZE = (224, 224)

# Data Transformations
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load Dataset
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)


# Define the ResNet model
model = models.resnet50(pretrained=True)  # Load ResNet50 with pre-trained weights

# Modify the fully connected layer to match the number of classes
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to the specified device
model = model.to(DEVICE)


# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# Function to Train One Epoch
def train_epoch(model, train_loader, criterion, device, optimizer):
    model.train()  # Set model to training mode

    epoch_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(inputs)  # Raw outputs (logits)
        loss = criterion(outputs, labels)

        # Backpropagation and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)  # Calculate average loss for the epoch
    return avg_loss


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


# device=torch.device("cuda:0")
device = torch.device("cpu")
# Use Subset for memory-efficient dataset management
train_dataset = Subset(train_dataset, range(trainval_split))
val_dataset = Subset(val_dataset, range(val_start, val_end))
test_dataset = train_dataset

# Define dataloaders
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=batchsize, shuffle=True),
    "val": DataLoader(val_dataset, batch_size=batchsize, shuffle=False),
    "test": DataLoader(test_dataset, batch_size=batchsize, shuffle=False),
}

best_valid_loss = float("inf")

for epoch in range(EPOCHS):

    start_time = time.monotonic()

    # Train the model for one epoch
    train_losses = train_epoch(
        model, dataloaders["train"], criterion, DEVICE, optimizer
    )
    train_loss = torch.mean(torch.tensor(train_losses)).item()  # Average training loss

    valid_accuracy, valid_losses = evaluate(
        model, dataloaders["val"], criterion, DEVICE
    )
    valid_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None

    # Save the model if validation loss improves
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "ResNet50_birds_model.pt")

    end_time = time.monotonic()

    # Calculate epoch duration
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    # Print epoch statistics
    print(f"Epoch: {epoch + 1:02} | Epoch Time: {int(epoch_mins)}m {int(epoch_secs)}s")
    print(f"\tTrain Loss: {train_loss:.3f}")
    print(
        f"\tValid Loss: {valid_loss:.3f} | Valid Accuracy: {valid_accuracy * 100:.2f}%"
    )

# Save the Model
torch.save(model.state_dict(), f"models/cnn_model.pth")
print("Model saved!")

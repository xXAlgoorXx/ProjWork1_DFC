import clip
import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
from Finetune_utils import CandolleDataset, CLIPFineTuner,getTexttokens

# Own modules
sys.path.append("/home/lukasschoepf/Documents/ProjWork1_DFC")

import folderManagment.pathsToFolders as ptf #Controlls all paths

# OpenAI CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", jit=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

files = os.listdir(ptf.Dataset5Patch)
# Split dataset into training and validation sets
train_size = int(0.6 * len(files))
val_size = int(0.2 * len(files))
test_size = int(0.2 * len(files))
train_dataset, val_dataset, test_dataset = random_split(files, [train_size, val_size,test_size])

trainData = CandolleDataset(preprocess)
trainData.loadList(train_dataset)

valData = CandolleDataset(preprocess)
valData.loadList(val_dataset)

testData = CandolleDataset(preprocess)
testData.loadList(test_dataset)

# Create DataLoader for training and validation sets
train_loader = DataLoader(trainData, batch_size = 1, shuffle=True)
val_loader = DataLoader(valData, batch_size=1, shuffle=False)
test_loader = DataLoader(testData, batch_size=1, shuffle=False)



# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
num_classes = 5
model_ft = CLIPFineTuner(model, num_classes).to(device)
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)




# Number of epochs for training
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    model_ft.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the current epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")  # Initialize progress bar
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)  # Move images and labels to the device (GPU or CPU)
        optimizer.zero_grad()  # Clear the gradients of all optimized variables
        outputs = model_ft(images)  # Forward pass: compute predicted outputs by passing inputs to the model
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)
        
        running_loss += loss.item()  # Update running loss
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")  # Update progress bar with current loss

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')  # Print average loss for the epoch

    # Validation
    model_ft.eval()  # Set the model to evaluation mode
    correct = 0  # Initialize correct predictions counter
    total = 0  # Initialize total samples counter
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the device
            outputs = model_ft(images)  # Forward pass: compute predicted outputs by passing inputs to the model
            _, predicted = torch.max(outputs.data, 1)  # Get the class label with the highest probability
            total += labels.size(0)  # Update total samples
            correct += (predicted == labels).sum().item()  # Update correct predictions

    print(f'Validation Accuracy: {100 * correct / total}%')  # Print validation accuracy for the epoch

# Save the fine-tuned model
torch.save(model_ft.state_dict(), 'clip_finetuned.pth')  # Save the model's state dictionary

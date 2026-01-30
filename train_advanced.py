import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import copy
import time

# CONFIG
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS_HEAD = 5   # Phase 1: Train head only
EPOCHS_FINE = 10  # Phase 2: Fine-tune backbone
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Training on device: {DEVICE}")
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),       # Zoom in on parts of the leaf
        transforms.RandomHorizontalFlip(),       # Flip left/right
        transforms.RandomRotation(15),           # Rotate slightly
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Change lighting
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root='dataset', transform=train_transforms)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")

    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    def train_one_epoch(model, loader, optimizer, is_train=True):
        if is_train:
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        corrects = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_train):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                if is_train:
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = corrects.double() / len(loader.dataset)
        
        return epoch_loss, epoch_acc

    print("\n--- Phase 1: Training Head Only ---")
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS_HEAD):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, is_train=True)
        val_loss, val_acc = train_one_epoch(model, val_loader, optimizer, is_train=False)
        
        print(f"Epoch {epoch+1}/{EPOCHS_HEAD} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    print("\n--- Phase 2: Fine-Tuning Backbone ---")
    
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ])
    
    for epoch in range(EPOCHS_FINE):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, is_train=True)
        val_loss, val_acc = train_one_epoch(model, val_loader, optimizer, is_train=False)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS_FINE} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'plant_doctor_model_v2.pth')
    print("Saved best model to plant_doctor_model_v2.pth")

if __name__ == '__main__':
    main()
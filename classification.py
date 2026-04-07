import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from DTU_ADLCV_attention_surgeon_grp10.utils import load_model
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

'''
Done:
- Open a dataset (CIFAR-10)
- Linear Probing for Classification task:
    * Model created (Dinov2 frozen + Linear head for classification)
    * Training the head on CIFAR-10 TODO : CLEAN TRAINING with loss function etc
    * Retrieve the 144 heads loacted by 0-11 blocks of attention with each attn.num_heads=12 (need to reshaape the weights of the qkv tensor) (Look at Hook of Dinov2)

TODO:
- Download ImageNet-1k ? Look at Dino with registers
- Once trained, plot head activation (4 methods) while prediction (average attention entropy/ average attention distance / head importance via gradient-based attribution / activation magnitude)
- First Pruning with mask but then find a solution for dynamic physical pruning (Gate ?)

'''

def main():
    #-------------------DATASET
    base_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=base_transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=base_transform
    )

    train_loader = DataLoader(
        train_set, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2
    )

    test_loader = DataLoader(
        test_set, 
        batch_size=32, 
        shuffle=False, 
        num_workers=2
    )

    print(f"Images d'entraînement : {len(train_set)}") # 50000
    print(f"Images de test : {len(test_set)}") # 10000

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"{classes=}")

    #-------------------Linear Probing
    class DinoClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super(DinoClassifier, self).__init__()
            self.transformer = load_model(device)
            
            for param in self.transformer.parameters():
                param.requires_grad = False

            self.classifier = nn.Linear(768, num_classes)

        def forward(self, x):
            features = self.transformer(x)
            logits = self.classifier(features)
            return logits

    model = DinoClassifier(num_classes=10).to(device)
    # For saving the checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    #---------------Training Linear Probing (Comment this piece if weights saved)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3) # we train only the head

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs): 
        train_loss = 0.
        val_loss = 0.
        
        # Train
        model.train()
        for images, labels in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss/len(train_loader))

        # Val
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            
        val_losses.append(val_loss/len(test_loader))
        checkpoint_path = f"checkpoints/dino_classifier_epoch_{epoch+1}.pth"
    
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
        }, checkpoint_path)
        
        print(f"Checkpoint sauvé : {checkpoint_path}")

    # Plot Train/val curves
    os.makedirs("figure", exist_ok=True)
    plt.plot(range(1, num_epochs + 1), train_losses, label="train", marker = 'o')
    plt.plot(range(1, num_epochs + 1), val_losses, label = "val", marker = 's')
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.savefig("figure/train_val_curve.jpg")
    plt.close()

if __name__ == "__main__":
    #------------------HYPERPARAMETERS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    num_epochs = 30
    main()
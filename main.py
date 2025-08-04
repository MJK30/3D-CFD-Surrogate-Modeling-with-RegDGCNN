import torch
from torch.utils.data import DataLoader
import argparse
import os
from models.RegDGCNN import RegDGCNN
from utils.dataset import ShapeNetCFD
from train_model import train_one_epoch, evaluate

def main():
    split_dir = "E:/Python/3D_CFD_Simulations/ShapeNet Data"
    data_dir = os.path.join(split_dir, "Processed_Data")
    train_txt = os.path.join(split_dir, "train.txt")
    test_txt = os.path.join(split_dir, "test.txt")
    save_dir = "./checkpoints"
    batch_size = 8
    lr = 0.01
    epochs = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = ShapeNetCFD(train_txt, data_dir=data_dir)
    test_dataset = ShapeNetCFD(test_txt, data_dir=data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"[INFO] Loaded {len(train_dataset)} training and {len(test_dataset)} test samples")

    # Model
    model = RegDGCNN().to(device)

    # Loss and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"regdgcnn_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[INFO] Saved model to {checkpoint_path}")
    
    
    
if __name__ == "__main__":
    main()
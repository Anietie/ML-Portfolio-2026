"""
Training loop - supervised learning from game tree data
Uses the 4-head model architecture to learn move policy, value, capture, and next state
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model.model import GameNet, GameLoss, GameConfig
from data.pipeline import make_loaders
import time

# training settings
CSV_PATH = "data/test_dataset.csv"
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # load data
    train_loader, val_loader = make_loaders(CSV_PATH, batch_size=BATCH_SIZE)
    
    # setup model and optimizer
    cfg = GameConfig()
    model = GameNet(cfg).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = GameLoss().to(DEVICE)
    
    # logging
    writer = SummaryWriter("runs/swin_v1")
    print(f"Training on {DEVICE} with {model.count_parameters():,} parameters...")

    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            targets = {k: v.to(DEVICE) for k, v in targets.items()}
            
            # forward pass
            preds = model(inputs['board'], inputs['turn'], inputs['context'])
            loss, components = criterion(preds, targets)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                for name, val in components.items():
                    writer.add_scalar(f"Loss/{name}", val, epoch * len(train_loader) + batch_idx)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                targets = {k: v.to(DEVICE) for k, v in targets.items()}
                preds = model(inputs['board'], inputs['turn'], inputs['context'])
                l, _ = criterion(preds, targets)
                val_loss += l.item()
        
        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start_time
        print(f"--- Epoch {epoch} Complete | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.1f}s ---")
        
        # save checkpoint
        torch.save(model.state_dict(), f"checkpoints/swin_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()
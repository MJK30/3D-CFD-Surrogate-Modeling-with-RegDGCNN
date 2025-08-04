import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        points = batch["points"].to(device)
        targets = batch["pressure"].to(device)

        optimizer.zero_grad()
        preds = model(points)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            targets = batch["pressure"].to(device)

            preds = model(points)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(loader)
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch in pbar:
        image_paths = batch["image_paths"]
        labels = batch["labels"].to(device)
        prompts = batch["prompts"]

        logits, _ = model(image_paths, prompts)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)

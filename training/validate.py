import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            image_paths = batch["image_paths"]
            labels = batch["labels"].to(device)
            prompts = batch["prompts"]

            logits, _ = model(image_paths, prompts)
            loss = criterion(logits, labels)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    auc = roc_auc_score(all_labels, all_probs)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

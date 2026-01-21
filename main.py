import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from models.chexagent_generative import CheXagentGenerative
from datasets.chexagent_dataset import CheXagentBinaryDataset, collate_fn
from training.train import train_epoch
from training.validate import validate


def main():

    # ==================== PATH CONFIG (EDIT HERE) ====================
    base_dir = "D:\panacea\Vinbig main\Vinbig"
    csv_path = "D:\panacea\combined.csv"

    # ==================== TRAINING CONFIG ====================
    batch_size = 1
    num_epochs = 10
    learning_rate = 1e-5
    warmup_epochs = 1
    train_split = 0.8
    num_workers = 2
    random_seed = 42
    save_dir = "checkpoints"

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(random_seed)

    # ==================== DEVICE ====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==================== LOAD CSV ====================
    import pandas as pd
    df = pd.read_csv(csv_path)

    image_ids = df["image_id"].tolist()
    labels_str = df["label"].tolist()

    label_map = {"normal": 0, "abnormal": 1}
    labels = [label_map[l.lower()] for l in labels_str]

    # ==================== MATCH IMAGES ====================
    available_images = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_id = os.path.splitext(f)[0]
                available_images[img_id] = os.path.join(root, f)

    image_paths, filtered_labels = [], []
    for img_id, label in zip(image_ids, labels):
        img_id = os.path.splitext(os.path.basename(img_id))[0]
        if img_id in available_images:
            image_paths.append(available_images[img_id])
            filtered_labels.append(label)

    print(f"Total matched images: {len(image_paths)}")

    # ==================== DATASET ====================
    dataset = CheXagentBinaryDataset(image_paths, filtered_labels)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # ==================== MODEL ====================
    model = CheXagentGenerative().to(device)

    print("\nTrainable parameters:")
    print(model.get_trainable_parameters())

    # ==================== OPTIMIZER ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # ==================== TRAINING LOOP ====================
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        val_metrics = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}\n"
            f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}"
        )

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import Dataset


class CheXagentBinaryDataset(Dataset):
    def __init__(self, image_paths, labels, prompts=None):
        self.image_paths = image_paths
        self.labels = labels
        self.prompts = prompts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {
            "image_path": self.image_paths[idx],
            "label": self.labels[idx],
            "prompt": None if self.prompts is None else self.prompts[idx]
        }


def collate_fn(batch):
    image_paths = [b["image_path"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    prompts = [b["prompt"] for b in batch]

    if all(p is None for p in prompts):
        prompts = None

    return {
        "image_paths": image_paths,
        "labels": labels,
        "prompts": prompts
    }

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, IntxWeightOnlyConfig
from torchao.quantization.qat import QATConfig, QATStep
from datasets import load_dataset
from collections import defaultdict
import random
import sys

DATASET_SLUG = "zh-plus/tiny-imagenet"
MODEL_DIR = "models"
NUM_CLASSES = 200

class _SegmentProcessor:
    def __init__(self, mean, std):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def __call__(self, batch):
        images = [
            self.transform(img.convert("RGB"))
            for img in batch["image"]
        ]
            
        return {
            "image": images,
            "label": batch["label"]
        }

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
            
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
    return device

def setup_dataloaders():
    full_train_dataset = load_dataset(DATASET_SLUG, split="train")
    val_dataset = load_dataset(DATASET_SLUG, split="valid")
    
    test_dataset, train_dataset = _gen_subset_from_dataset(full_train_dataset)
    
    train_mean, train_std = _compute_dataset_stats(train_dataset)
    print(f"Dataset statistics: mean={train_mean.tolist()} std={train_std.tolist()}")
    
    processor = _SegmentProcessor(mean=train_mean, std=train_std)
    train_dataset.set_transform(processor)
    val_dataset.set_transform(processor)
    test_dataset.set_transform(processor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True) # pyright: ignore[reportArgumentType]
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True) # pyright: ignore[reportArgumentType]
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True) # pyright: ignore[reportArgumentType]
    
    return (train_dataloader, val_dataloader, test_dataloader)

def get_ptq_modes():
    modes = {
        "int8": Int8WeightOnlyConfig(version=2)
    }
    
    if sys.platform.startswith("linux"):
        modes["int4"] = Int4WeightOnlyConfig() # pyright: ignore[reportArgumentType]
    else:
        print("Skipping int4: platform is not linux")
    
    return modes

def get_qat_modes():
    modes = {
        "int8": (
            QATConfig(IntxWeightOnlyConfig(torch.int8), step=QATStep.PREPARE),
            QATConfig(IntxWeightOnlyConfig(torch.int8), step=QATStep.CONVERT)
        )
    }
        
    if sys.platform.startswith("linux"):
        modes["int4"] = (
            QATConfig(Int4WeightOnlyConfig(), step=QATStep.PREPARE),
            QATConfig(Int4WeightOnlyConfig(), step=QATStep.CONVERT)
        )
    else:
        print("Skipping int4: platform is not linux")
    
    return modes

def _gen_subset_from_dataset(dataset, num=50, seed=0):
    rng = random.Random(seed)
    indices_by_label = defaultdict(list)
        
    for id, label in enumerate(dataset["label"]):
        indices_by_label[label].append(id)
        
    subset_indices = set()
    for label, indices in indices_by_label.items():
        if len(indices) < num:
            raise ValueError(f"Class {label} has fewer than {num} samples")
        subset_indices.update(rng.sample(indices, num))
        
    keep_indices = [i for i in range(len(dataset)) if i not in subset_indices]
    return (dataset.select(subset_indices), dataset.select(keep_indices))

def _compute_dataset_stats(dataset):
    def process(i):
        images = i["image"]
        new_images = [transforms.ToTensor()(
            img.convert("RGB")
            ) for img in images]
            
        return {"image": new_images}
        
    dataset.set_transform(process)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0
        
    for batch in loader:
        images = batch["image"]
        b, _c, h, w = images.shape
            
        num_pixels += b * h * w
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
        
    mean = channel_sum / num_pixels
    std = (channel_sq_sum / num_pixels - mean ** 2).sqrt()
        
    dataset.reset_format()
    return (mean, std)

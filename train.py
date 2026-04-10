import torch
import torch.nn as nn
import torchvision
from torchao.quantization import quantize_
import copy
import os
import math

from common import MODEL_DIR, NUM_CLASSES, setup_device, setup_dataloaders, get_ptq_modes, get_qat_modes

EPOCH_COUNT = 50

# On Linux, install package fbgemm-gpu-genai

def train(device, model, optimizer, train_dataloader, loss_fn, scalar, half=False):
    running_loss = 0.0
    num_samples = 0
    
    for batch in train_dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.autocast(device.type, enabled=half, dtype=torch.float16):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        # Adjust weights
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size
    
    return running_loss / num_samples

@torch.no_grad()
def validate(device, model, val_dataloader, loss_fn, half=False):
    running_loss = 0.0
    num_samples = 0
    
    for batch in val_dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        with torch.autocast(device.type, enabled=half, dtype=torch.float16):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size
    
    return running_loss / num_samples

def create_trained_model(device, train_dataloader, val_dataloader, save_dir, half = False, qat_config = None):
    def save_model(model, name):
        if qat_config is None:
            torch.save(model.state_dict(), f"{save_dir}/{name}")
            return
        
        temp_model = copy.deepcopy(model)
        quantize_(temp_model, qat_config[1])
        torch.save(temp_model.state_dict(), f"{save_dir}/{name}")
    
    model = torchvision.models.resnet18(num_classes=NUM_CLASSES)
    model.to(device)
    
    if qat_config is not None:
        quantize_(model, qat_config[0], device=device)
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scalar = torch.GradScaler(device=device.type, enabled=half)
    
    best_loss = float("inf")
    for epoch in range(EPOCH_COUNT):
        print(f"Starting epoch {epoch + 1}/{EPOCH_COUNT}")
        
        model.train()
        train_loss = train(device, model, optimizer, train_dataloader, loss_fn, scalar, half)
        
        model.eval()
        val_loss = validate(device, model, val_dataloader, loss_fn, half)
        
        save_model(model, "last.pt")
        
        if not math.isnan(val_loss) and val_loss < best_loss:
            # Replace model
            save_model(model, "best.pt")
            best_loss = val_loss
            
        print(f"LOSS train={train_loss} val={val_loss}")
        
    print(f"Training complete, best validation loss={best_loss}")
    
    return model

def ptq_and_save(device, model):
    dir_path = f"{MODEL_DIR}/ptq"
    os.mkdir(dir_path)
    
    model.eval()
    modes = get_ptq_modes()
    for name, mode in modes.items():
        print(f"Quantization to {name}")
        
        save_path = f"{dir_path}/{name}.pt"
        new_model = copy.deepcopy(model)
        
        quantize_(new_model, mode, device=device)
        
        torch.save(new_model.state_dict(), save_path)
        print(f"Quantized to {name} at {save_path}")
    
    # Float-16 is only a cast
    def gen_and_write_f16():
        print(f"Quantization to float16")
        save_path = f"{dir_path}/float16.pt"
        new_model = copy.deepcopy(model)
        new_model = new_model.half()
        torch.save(new_model.state_dict(), save_path)
        print(f"Quantized to float16 at {save_path}")
    
    gen_and_write_f16()

def qat_and_save(device, train_dataloader, val_dataloader):
    dir_path = f"{MODEL_DIR}/qat"
    os.mkdir(dir_path)
    
    modes = get_qat_modes()
    for name, mode in modes.items():
        print(f"Training {name}")
        
        save_dir = f"{dir_path}/{name}"
        os.mkdir(save_dir)
        
        create_trained_model(device, train_dataloader, val_dataloader, save_dir, qat_config=mode)
        print(f"Quantized to {name} at {save_dir}")
    
    # Float-16 is only a cast
    def gen_and_write_f16():
        print(f"Training float16")
        save_dir = f"{dir_path}/float16"
        os.mkdir(save_dir)
        
        create_trained_model(device, train_dataloader, val_dataloader, save_dir, half=True)
        print(f"Quantized to float16 at {save_dir}")
    
    gen_and_write_f16()

def main():
    if os.path.exists(MODEL_DIR):
        raise RuntimeError("Model path exists")
    else:
        os.mkdir(MODEL_DIR)
    
    device = setup_device()
    
    print("Preparing dataset...")
    train_dataloader, val_dataloader, _ = setup_dataloaders()
    
    print("Training base model...")
    base_model = create_trained_model(device, train_dataloader, val_dataloader, MODEL_DIR)
    
    print("Training QAT models...")
    qat_and_save(device, train_dataloader, val_dataloader)
    
    print("Running PTQ...")
    ptq_and_save(device, base_model)
    
    print("Complete!")

if __name__ == "__main__":
    main()

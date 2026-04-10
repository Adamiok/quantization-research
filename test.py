import numpy as np
import torch
import torchvision
from torchao.quantization import quantize_
import psutil
from contextlib import nullcontext
import dataclasses
import gc
import os
import time
import json

from common import MODEL_DIR, NUM_CLASSES, setup_device, setup_dataloaders, get_ptq_modes, get_qat_modes

BEST_NAME = "best.pt"
RESULTS_FILE = "results.json"
BENCHMARK_WARMUP = 2
BENCHMARK_REPEAT = 10_000
THROUGHOUT_REPEAT = 5

# On Linux, install package fbgemm-gpu-genai

@dataclasses.dataclass
class TestResult:
    accuracy: float
    bench: dict[str, float]

class DataclassJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        
        return super().default(o)

def _get_process_memory():
    return psutil.Process().memory_info().rss / (1024 ** 2)

def _load_model(device, path, quantization = None, half = False):
    model = torchvision.models.resnet18(num_classes=NUM_CLASSES)
    
    if isinstance(quantization, (list, tuple)):
        quantize_(model, quantization[0], device=device)
        quantize_(model, quantization[1], device=device)
    elif quantization is not None:
        quantize_(model, quantization, device=device)
    if half:
        model.half()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.to(device)
    
    return model

@torch.inference_mode()
def _test(device, model, dataloader, autocast: torch.dtype | None = None, half = False):
    gpu = device.type == "cuda"
    cast = torch.autocast("cuda", dtype=autocast) if gpu and autocast is not None else nullcontext()
    model.eval()
    
    successful_samples = 0
    num_samples = 0
    
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        if half:
            images = images.half()
        
        with cast:
            outputs = model(images)
        predictions = outputs.argmax(dim=1)
        
        successful_samples += (predictions == labels).sum().item()
        num_samples += labels.size(0)
    
    accuracy = successful_samples / num_samples
    return accuracy

@torch.inference_mode()
def _bench(device, model, dataloader, autocast: torch.dtype | None = None, half = False):
    gpu = device.type == "cuda"
    cast = torch.autocast("cuda", dtype=autocast) if gpu and autocast is not None else nullcontext()
    model.eval()
    gc.disable()
    
    it = iter(dataloader)
    next(it) # Throw away
    images = next(it)["image"].to(device)
    if half:
        images = images.half()
    
    if gpu:
        torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(BENCHMARK_WARMUP):
        with cast:
            model(images)
    
    if gpu:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # MiB
    else:
        start_memory = _get_process_memory() # MiB
    
    # Latency
    latencies = []
    for _ in range(BENCHMARK_REPEAT):
        if gpu:
            torch.cuda.synchronize()
        
        start = time.time_ns()
        with cast:
            model(images)
        if gpu:
            torch.cuda.synchronize()
        
        end = time.time_ns()
        latencies.append((end - start) * 1e-6) # ms
    
    if gpu:
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2) - start_memory
    else:
        memory_used = _get_process_memory() - start_memory
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    # Throughput
    if gpu:
        torch.cuda.synchronize()
    
    throughputs = []
    for i in range(THROUGHOUT_REPEAT + 1):
        num_images = 0
        start = time.time()
        
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            if half:
                images = images.half()
            
            with cast:
                model(images)
            num_images += images.size(0)
        
        if gpu:
            torch.cuda.synchronize()
        end = time.time()
        throughput = num_images / (end - start)
        
        if i > 0:
            # Skip first (warmup) batch
            throughputs.append(throughput)
    
    avg_throughput = np.mean(throughputs)
    std_throughput = np.std(throughputs)
    
    gc.collect()
    gc.enable()
    
    return {
        "avg_latency": avg_latency,
        "std_latency": std_latency,
        "memory_used": memory_used,
        "avg_throughput": avg_throughput,
        "std_throughput": std_throughput
    }

def test_base(device, dataloader):
    path = f"{MODEL_DIR}/{BEST_NAME}"
    model = _load_model(device, path)
    
    accuracy = _test(device, model, dataloader)
    bench = _bench(device, model, dataloader)
    return TestResult(accuracy, bench)

def test_ptq(device, dataloader):
    modes = get_ptq_modes()
    results = {}
        
    for name, mode in modes.items():
        try:
            path = f"{MODEL_DIR}/ptq/{name}.pt"
            model = _load_model(device, path, mode)
            cast = torch.bfloat16 if name == "int4" else None
            
            accuracy = _test(device, model, dataloader, cast)
            bench = _bench(device, model, dataloader, cast)
            results[name] = TestResult(accuracy, bench)
        except FileNotFoundError:
            print(f"ERROR: PTQ {name} not found")
            results[name] = None
    
    # Float 16
    try:
        path = f"{MODEL_DIR}/ptq/float16.pt"
        model = _load_model(device, path, half=True)
        
        accuracy = _test(device, model, dataloader, half=True)
        bench = _bench(device, model, dataloader, half=True)
        results["float16"] = TestResult(accuracy, bench)
    except FileNotFoundError:
        print(f"ERROR: PTQ float16 not found")
        results["float16"] = None
        
    return results

def test_qat(device, dataloader):
    modes = get_qat_modes()
    results = {}
        
    for name, mode in modes.items():
        try:
            path = f"{MODEL_DIR}/qat/{name}/{BEST_NAME}"
            model = _load_model(device, path, mode)
            cast = torch.bfloat16 if name == "int4" else None
            
            accuracy = _test(device, model, dataloader, cast)
            bench = _bench(device, model, dataloader, cast)
            results[name] = TestResult(accuracy, bench)
        except FileNotFoundError:
            print(f"ERROR: QAT {name} not found")
            results[name] = None
        
    # Float 16
    try:
        path = f"{MODEL_DIR}/qat/float16/{BEST_NAME}"
        model = _load_model(device, path)
        
        accuracy = _test(device, model, dataloader, torch.float16)
        bench = _bench(device, model, dataloader, torch.float16)
        results["float16"] = TestResult(accuracy, bench)
    except FileNotFoundError:
        print(f"ERROR: PTQ float16 not found")
        results["float16"] = None
        
    return results

def main():
    device = setup_device()
    
    print("Preparing dataset...")
    _, _, test_dataloader = setup_dataloaders()
    results = {}
    
    print("Testing base model...")
    results["base"] = test_base(device, test_dataloader)
    
    print("Testing PTQ...")
    results["ptq"] = test_ptq(device, test_dataloader)
    
    print("Testing QAT...")
    results["qat"] = test_qat(device, test_dataloader)
    
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, cls=DataclassJsonEncoder, indent=4)
    except Exception:
        os.unlink(RESULTS_FILE)
        raise
    
    print(f"Completed! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()

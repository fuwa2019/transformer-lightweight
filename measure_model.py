import time
import os
import torch
import torch.nn as nn

def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

device = get_device()
print("Current device:", device)

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        return self.net(x)
    
model = TinyModel().to(device)
model.eval()

num_params = sum(p.numel() * p.element_size() for p in model.parameters())
print("Number of parameters:", num_params)

param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
param_size_mb = param_size_bytes / 1024 / 1024
print(f"Estimated parameter size: {param_size_mb:.2f} MB")

save_path = "tiny_model_fp32.pt"
torch.save(model.state_dict(), save_path)
file_size_mb = os.path.getsize(save_path) / 1024 / 1024
print(f"Saved model file size: {file_size_mb:.2f} MB")

x = torch.randn(32, 1024).to(device)

with torch.no_grad():
    for _ in range(10):
        y = model(x)

if device.type == "mps":
    torch.mps.synchronize()

start = time.perf_counter()

with torch.no_grad():
    for _ in range(100):
        y = model(x)

if device.type == "mps":
    torch.mps.synchronize()

end = time.perf_counter()

avg_latency_ms = (end - start) / 100 * 1000
print(f"Averge inference latency: {avg_latency_ms:.3f} ms")

print("Output shape:", y.shape)
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(3, 3).cuda()
y = x @ x

print(y)
print("Device:", y.device)
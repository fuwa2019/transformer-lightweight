import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)
x_fp16 = x_fp32.to(torch.float16)
x_int8 = x_fp32.to(torch.int8)

print(x_fp32.element_size())  # 4 bytes
print(x_fp16.element_size())  # 2 bytes
print(x_int8.element_size())  # 1 byte
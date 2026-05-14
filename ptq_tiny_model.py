import torch
import torch.nn as nn

def quantize_tensor_uint8(x, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1

    x_min = x.min()
    x_max = x.max()

    scale = (x_max - x_min) / (qmax - qmin)

    if scale.item() == 0:
        scale = torch.tensor(1.0, device=x.device)

    zero_point = qmin - torch.round(x_min / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    x_int = torch.round(x / scale + zero_point)
    x_int = torch.clamp(x_int, qmin, qmax).to(torch.uint8)

    x_dequant = scale * (x_int.float() - zero_point)

    return x_int, x_dequant, scale, zero_point

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

model_fp32 = TinyModel()
model_fp32.eval()

model_ptq = TinyModel()
model_ptq.load_state_dict(model_fp32.state_dict())
model_ptq.eval()

x = torch.randn(32, 1024)

with torch.no_grad():
    y_fp32 = model_fp32(x)

quant_info = {}

with torch.no_grad():
    for name, module in model_ptq.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data

            w_int, w_dequant, scale, zero_point = quantize_tensor_uint8(w)

            module.weight.data = w_dequant

            quant_info[name] = {
                "scale": scale.item(),
                "zero_point": zero_point.item(),
                "original_weight_size_MB": w.numel() * w.element_size() / 1024 / 1024,
                "int8_weight_size_MB": w_int.numel() * w_int.element_size() / 1024 / 1024,
            }

with torch.no_grad():
    y_ptq = model_ptq(x)

mae = torch.mean(torch.abs(y_fp32 - y_ptq))
max_error = torch.max(torch.abs(y_fp32 - y_ptq))

print("Output mean absolute error:", mae.item())
print("Output max absolute error:", max_error.item())

print("\nQuantization info for Linear layers:")
for layer_name, info in quant_info.items():
    print(f"\nLayer: {layer_name}")
    print(f"  scale: {info['scale']}")
    print(f"  zero_point: {info['zero_point']}")
    print(f"  original FP32 weight size: {info['original_weight_size_MB']:.4f} MB")
    print(f"  INT8 weight size: {info['int8_weight_size_MB']:.4f} MB")
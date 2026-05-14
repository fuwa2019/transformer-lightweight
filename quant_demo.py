import torch


def quantize_tensor(x, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1

    x_min = x.min()
    x_max = x.max()

    scale = (x_max - x_min) / (qmax - qmin)

    if scale.item() == 0:
        scale = torch.tensor(1.0)

    zero_point = qmin - torch.round(x_min / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    x_int = torch.round(x / scale + zero_point)
    x_int = torch.clamp(x_int, qmin, qmax).to(torch.uint8)

    x_dequant = scale * (x_int.float() - zero_point)

    return x_int, x_dequant, scale, zero_point


x = torch.tensor([-1.0, -0.5, 0.0, 0.3, 0.8, 1.2], dtype=torch.float32)

x_int, x_dequant, scale, zero_point = quantize_tensor(x, num_bits=8)

print("Original x:")
print(x)

print("\nQuantized int:")
print(x_int)

print("\nDequantized x:")
print(x_dequant)

print("\nScale:")
print(scale.item())

print("\nZero point:")
print(zero_point.item())

print("\nAbsolute error:")
print(torch.abs(x - x_dequant))
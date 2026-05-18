import torch

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

def run_case(name, x, clip_min=None, clip_max=None):
    print("=" * 70)
    print(name)
    print("=" * 70)

    if clip_min is not None and clip_max is not None:
        x_used = torch.clamp(x, clip_min, clip_max)
        print("Clipping range:", clip_min, "to", clip_max)
    else:
        x_used = x
        print("No clipping")

    x_int, x_dequant, scale, zero_point = quantize_tensor_uint8(x_used)

    abs_error_all = torch.abs(x - x_dequant)
    abs_error_normal = torch.abs(x[:-1] - x_dequant[:-1])

    print("\nOriginal x:")
    print(x)

    print("\nUsed for quantization:")
    print(x_used)

    print("\nQuantized int:")
    print(x_int)

    print("\nDequantized x:")
    print(x_dequant)

    print("\nScale:")
    print(scale.item())

    print("\nZero point:")
    print(zero_point.item())

    print("\nMean absolute error for normal values, excluding outlier:")
    print(abs_error_normal.mean().item())

    print("\nMax absolute error for normal values, excluding outlier:")
    print(abs_error_normal.max().item())

    print()

x = torch.tensor(
    [-1.0, -0.5, 0.0, 0.3, 0.8, 1.0, 20.0],
    dtype=torch.float32
)

run_case("Case 1: Without clipping", x)
run_case("Case 2: With clipping", x, clip_min=-1.0, clip_max=1.0)
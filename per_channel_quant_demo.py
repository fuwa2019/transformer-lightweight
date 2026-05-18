import torch


def quantize_uint8_affine(x, reduce_dim=None, num_bits=8):
    """
    uint8 affine quantization.

    reduce_dim=None:
        整个 tensor 共用一个 scale / zero_point，也就是 per-tensor quantization。

    reduce_dim=1:
        对二维矩阵按“行”量化。
        对形状 [out_features, in_features] 的 Linear 权重来说，
        每一行是一个输出通道，所以这相当于 per-channel quantization。
    """
    qmin = 0
    qmax = 2 ** num_bits - 1

    if reduce_dim is None:
        x_min = x.min()
        x_max = x.max()
    else:
        x_min = x.min(dim=reduce_dim, keepdim=True).values
        x_max = x.max(dim=reduce_dim, keepdim=True).values

    # 关键修正：
    # 量化范围要包含 0。
    # 否则像 [1, 2, 3, 4] 这种全正数行，
    # zero_point 会变成负数，再被 clamp 到 0，导致最大值被截断。
    x_min = torch.minimum(x_min, torch.zeros_like(x_min))
    x_max = torch.maximum(x_max, torch.zeros_like(x_max))

    scale = (x_max - x_min) / (qmax - qmin)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    zero_point = qmin - torch.round(x_min / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    x_int = torch.round(x / scale + zero_point)
    x_int = torch.clamp(x_int, qmin, qmax).to(torch.uint8)

    x_dequant = scale * (x_int.float() - zero_point)

    return x_int, x_dequant, scale, zero_point


def report(name, x, x_int, x_dequant, scale, zero_point):
    abs_error = torch.abs(x - x_dequant)

    print("=" * 70)
    print(name)
    print("=" * 70)

    print("Quantized int:")
    print(x_int)

    print("\nDequantized x:")
    print(x_dequant)

    print("\nScale:")
    print(scale)

    print("\nZero point:")
    print(zero_point)

    print("\nAbsolute error:")
    print(abs_error)

    print("\nMean absolute error:")
    print(abs_error.mean().item())

    print("\nMax absolute error:")
    print(abs_error.max().item())

    print()


def main():
    x = torch.tensor([
        [0.01, 0.02, 0.03, 0.04],
        [1.0, 2.0, 3.0, 4.0],
        [-20.0, -10.0, 0.0, 10.0],
    ], dtype=torch.float32)

    print("Original x:")
    print(x)
    print()

    # 1. per-tensor：整张矩阵共用一个 scale / zero_point
    x_int_tensor, x_dequant_tensor, scale_tensor, zp_tensor = quantize_uint8_affine(
        x,
        reduce_dim=None,
        num_bits=8
    )

    report(
        "Per-tensor quantization",
        x,
        x_int_tensor,
        x_dequant_tensor,
        scale_tensor,
        zp_tensor
    )

    # 2. per-channel：每一行单独一个 scale / zero_point
    # 对二维矩阵来说，reduce_dim=1 表示每一行自己统计 min/max
    x_int_channel, x_dequant_channel, scale_channel, zp_channel = quantize_uint8_affine(
        x,
        reduce_dim=1,
        num_bits=8
    )

    report(
        "Per-channel quantization",
        x,
        x_int_channel,
        x_dequant_channel,
        scale_channel,
        zp_channel
    )


if __name__ == "__main__":
    main()
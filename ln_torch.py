import torch
from torch.autograd import Function
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import triton

class LayerNormFunction(Function):

    @staticmethod
    def forward(ctx, x, scale, shift, eps):
        """
        Performs the forward pass of layer normalization.

        Args:
            ctx: Context object to save information for backward computation.
            input (Tensor): Input tensor of any shape.
            weight (Tensor, optional): Learnable per-element scale parameter.
            bias (Tensor, optional): Learnable per-element shift parameter.
            normalized_shape (int or tuple): Shape over which to normalize.
            eps (float, optional): Small epsilon for numerical stability.

        Returns:
            Tensor: Normalized tensor with the same shape as input.
        """

        mean = x.mean(dim=-1)
        rstd = torch.rsqrt(x.var(dim=-1, unbiased=False) + eps)
        x_norm = (x - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)
        y = x_norm * scale + shift
        ctx.save_for_backward(x, mean, rstd, scale, shift)

        return y


    @staticmethod
    def backward(ctx, dy):
        """
        Performs the backward pass of layer normalization.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[Tensor, Tensor, Tensor, None, None]: Gradients with respect to input,
            weight, bias, and None for normalized_shape and eps.
        """
        x, mean, rstd, scale, shift = ctx.saved_tensors

        x_norm = (x - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)

        if scale is not None:
            weighted_dy = dy * scale
        else:
            weighted_dy = dy

        dx = (1.0 / x_norm.shape[-1]) * rstd.unsqueeze(-1) * (
            x_norm.shape[-1] * weighted_dy
            - weighted_dy.sum(dim=-1, keepdim=True)
            - x_norm * (weighted_dy * x_norm).sum(dim=-1, keepdim=True)
        )

        if scale is not None and ctx.needs_input_grad[1]:
            d_scale = (dy * x_norm).sum(0)
        if shift is not None and ctx.needs_input_grad[2]:
            d_shift = dy.sum(0)

        return dx, d_scale, d_shift, None



# Unit Test
def test_ln_function():
    torch.manual_seed(42)

    # Create random input
    input = torch.randn(4, 5, 6, requires_grad=True)

    # Define normalized_shape
    normalized_shape = (6,)

    # Create weight and bias
    layer_norm = torch.nn.LayerNorm(normalized_shape, eps=1e-5)
    layer_norm.weight.data = torch.randn(6)
    layer_norm.bias.data = torch.randn(6)

    # Using custom LayerNormFunction
    input_c = input.clone().detach().requires_grad_(True)

    output_custom = LayerNormFunction.apply(input_c, layer_norm.weight, layer_norm.bias, 1e-5)
    loss_custom = output_custom.sum()
    loss_custom.backward()

    # Store gradients from custom function
    grad_input_custom = input_c.grad.clone()
    grad_scale_custom = layer_norm.weight.grad.clone()
    grad_shift_custom = layer_norm.bias.grad.clone()

    # zero gradients
    input_c.grad.data.zero_()
    layer_norm.weight.grad.data.zero_()
    layer_norm.bias.grad.data.zero_()

    # Using built-in LayerNorm
    input_b = input.clone().detach().requires_grad_(True)
    output_builtin = layer_norm(input_b)
    loss_builtin = output_builtin.sum()
    loss_builtin.backward()

    # Store gradients from built-in LayerNorm
    grad_input_builtin = input_b.grad.clone()
    grad_scale_builtin = layer_norm.weight.grad.clone()
    grad_shift_builtin = layer_norm.bias.grad.clone()

    # Assertions to verify correctness
    print(torch.allclose(output_custom, output_builtin, atol=1e-6, rtol=1e-4))
    print(torch.allclose(grad_input_custom, grad_input_builtin, atol=1e-6, rtol=1e-4))
    print(torch.allclose(grad_scale_custom, grad_scale_builtin, atol=1e-6, rtol=1e-4))
    print(torch.allclose(grad_shift_custom, grad_shift_builtin, atol=1e-6, rtol=1e-4))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['custom', 'torch'],
        line_names=['custom', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='linear-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_ln(M, N, dtype, provider, mode='forward', eps=1e-5, device='cuda', rep=100):
    # create data
    x_shape = (M, N)
    norm = torch.nn.LayerNorm(N, eps=eps).to(device).to(dtype)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "custom":
            return LayerNormFunction.apply(x, norm.weight, norm.bias, eps)

        if provider == "torch":
            return norm(x)

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=rep)

    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=rep)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
    
if __name__ == "__main__":
  test_ln_function()
  bench_ln.run(save_path='.', print_data=True)
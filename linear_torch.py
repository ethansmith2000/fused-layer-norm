import torch
from torch.autograd import Function
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import triton

class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        # reshape to batch
        orig_shape = input.shape
        input = input.view(-1, input.shape[-1])
        ctx.save_for_backward(input, weight, bias)
        output = torch.nn.functional.linear(input, weight, bias)
        return output.view(*orig_shape[:-1], -1)


    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        orig_shape = grad_output.shape
        grad_output = grad_output.view(-1, grad_output.shape[-1])

        grad_input = torch.nn.functional.linear(grad_output, weight.t())
        grad_weight = grad_bias = None
        if ctx.needs_input_grad[-2]:
            grad_weight = torch.nn.functional.linear(grad_output.t(), input.t())
        if ctx.needs_input_grad[-1] and bias is not None:
            grad_bias = grad_output.sum(dim=0)

        return grad_input.view(*orig_shape[:-1], -1), grad_weight, grad_bias

    
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
        args={'M': 32, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_linear(M, N, dtype, provider, mode='forward', eps=1e-5, device='cuda', rep=100):
    # create data
    x_shape = (2, M, N)
    linear = torch.nn.Linear(N, N).to(device).to(dtype)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "custom":
            return LinearFunction.apply(x, linear.weight, linear.bias)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.linear(x, linear.weight, linear.bias)  # noqa: F811, E704

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


if __name__ == '__main__':
    # test_layer_norm(1152, 8192, torch.float16)
    bench_layer_norm.run(save_path='.', print_data=True)
import torch
from torch.autograd import Function
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class LayerNormLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, scale, shift, normalized_shape, weight, bias, eps=1e-5):
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

        # layer norm #
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        orig_shape = input.shape
        input = input.view(-1, input.shape[-1])

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        dims = tuple(-i for i in range(1, len(normalized_shape)+1))
        mean = input.mean(dim=dims, keepdim=True)
        var = input.var(dim=dims, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + eps)

        x_norm = (input - mean) * rstd
        y = x_norm * scale + shift

        # linear #
        ctx.save_for_backward(y, mean, rstd, scale, shift, weight, bias)
        output = torch.nn.functional.linear(y, weight, bias)

        output = output.view(*orig_shape[:-1], -1)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs the backward pass of layer normalization.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[Tensor, Tensor, Tensor, None, None]: Gradients with respect to input,
            weight, bias, and None for normalized_shape and eps.
        """
        y, mean, rstd, scale, shift, weight, bias = ctx.saved_tensors

        orig_shape = grad_output.shape

        # Gradients w.r.t. Linear layer parameters
        grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        # this is the one that progresses back to layer norm layer
        grad_input_linear = torch.nn.functional.linear(grad_output_reshaped, weight.t())
        grad_weight = grad_bias = grad_scale = grad_shift = None
        if ctx.needs_input_grad[-3]:
            grad_weight = torch.nn.functional.linear(grad_output_reshaped.t(), y.t())
        if ctx.needs_input_grad[-2] and bias is not None:
            grad_bias = grad_output_reshaped.sum(dim=0)

        # recompute 
        x_norm = (y - shift) / scale

        if scale is not None:
            weighted_grad_input_linear = grad_input_linear * scale
        else:
            weighted_grad_input_linear = grad_input_linear

        grad_input = (1.0 / x_norm.shape[-1]) * rstd * (
            x_norm.shape[-1] * weighted_grad_input_linear
            - weighted_grad_input_linear.sum(dim=-1, keepdim=True)
            - x_norm * (weighted_grad_input_linear * x_norm).sum(dim=-1, keepdim=True)
        )

        if weight is not None and ctx.needs_input_grad[1]:
            grad_scale = (grad_input_linear * x_norm).sum(0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_shift = grad_input_linear.sum(0)

        grad_input = grad_input.view(*orig_shape[:-1], -1)

        # Return gradients for input, weight, bias, and None for normalized_shape and eps
        return grad_input, grad_scale, grad_shift, None, grad_weight, grad_bias, None


# Unit Test
def test_layer_norm_function():
    torch.manual_seed(42)

    # Create random input
    input = torch.randn(4, 5, 6, requires_grad=True)

    # Define normalized_shape
    normalized_shape = (6,)

    # Create weight and bias
    layer_norm = torch.nn.LayerNorm(normalized_shape, eps=1e-5)
    layer_norm.weight.data = torch.randn(6)
    layer_norm.bias.data = torch.randn(6)
    linear = torch.nn.Linear(6,6)

    # Using custom LayerNormFunction
    input_c = input.clone().detach().requires_grad_(True)

    output_custom = LayerNormLinearFunction.apply(input_c, layer_norm.weight, layer_norm.bias, normalized_shape, linear.weight, linear.bias, layer_norm.eps)
    loss_custom = output_custom.sum()
    loss_custom.backward()

    # Store gradients from custom function
    grad_input_custom = input_c.grad.clone()
    grad_scale_custom = layer_norm.weight.grad.clone()
    grad_shift_custom = layer_norm.bias.grad.clone()
    grad_weight_custom = linear.weight.grad.clone()
    grad_bias_custom = linear.bias.grad.clone()

    # zero gradients
    input_c.grad.data.zero_()
    layer_norm.weight.grad.data.zero_()
    layer_norm.bias.grad.data.zero_()
    linear.weight.grad.data.zero_()
    linear.bias.grad.data.zero_()

    # Using built-in LayerNorm
    input_b = input.clone().detach().requires_grad_(True)
    out_norm = layer_norm(input_b)
    output_builtin = linear(out_norm)
    loss_builtin = output_builtin.sum()
    loss_builtin.backward()

    # Store gradients from built-in LayerNorm
    grad_input_builtin = input_b.grad.clone()
    grad_scale_builtin = layer_norm.weight.grad.clone()
    grad_shift_builtin = layer_norm.bias.grad.clone()
    grad_weight_builtin = linear.weight.grad.clone()
    grad_bias_builtin = linear.bias.grad.clone()

    # Assertions to verify correctness
    print(torch.allclose(output_custom, output_builtin, atol=1e-6, rtol=1e-4))
    print(torch.allclose(grad_input_custom, grad_input_builtin, atol=1e-6, rtol=1e-4))
    print(torch.allclose(grad_scale_custom, grad_scale_builtin, atol=1e-6, rtol=1e-4))
    print(torch.allclose(grad_shift_custom, grad_shift_builtin, atol=1e-6, rtol=1e-4))
    print(torch.allclose(grad_weight_custom, grad_weight_builtin, atol=1e-6, rtol=1e-4))
    print(torch.allclose(grad_bias_custom, grad_bias_builtin, atol=1e-6, rtol=1e-4))



def benchmark_layer_norm_linear(
    batches=(128,), 
    dims=([i*512 for i in range(2, 16)]),
):

    times_torch = []
    times_custom = []
    pbar = tqdm(total=len(batches)*len(dims))
    for batch in batches:
        for dim in dims:
            torch.manual_seed(42)

            # Create random input
            input = torch.randn(batch, dim, requires_grad=True)
            layer_norm = torch.nn.LayerNorm(dim, eps=1e-5)
            linear = torch.nn.Linear(dim, dim)

            # torch
            start = time.time()
            input_b = input.clone().detach().requires_grad_(True)
            out_norm = layer_norm(input_b)
            output_builtin = linear(out_norm)
            loss_builtin = output_builtin.sum()
            loss_builtin.backward()
            end = time.time()
            times_torch.append(end-start)

            # custom
            start = time.time()
            input_c = input.clone().detach().requires_grad_(True)
            output_custom = LayerNormLinearFunction.apply(input_c, layer_norm.weight, layer_norm.bias, dim, linear.weight, linear.bias, layer_norm.eps)
            loss_custom = output_custom.sum()
            loss_custom.backward()
            end = time.time()
            times_custom.append(end-start)
            pbar.update(1)

    
    plt.figure(figsize=(10, 5))
    plt.plot(times_torch, label='torch')
    plt.plot(times_custom, label='custom')
    plt.xlabel('dim')
    plt.ylabel('Time (s)')
    plt.legend()
    
if __name__ == "__main__":
  test_layer_norm_function()
  benchmark_layer_norm_linear()

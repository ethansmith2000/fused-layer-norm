import torch

import triton
import triton.language as tl
import torch.nn.functional as F
from triton.language.math import rsqrt


try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False


@triton.jit
def _layer_norm_fwd_fused(
    X_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_ptr,  # pointer to input, shape (n_rows, n_cols)
    W_ptr,  # pointer to weights, shape (n_cols,)
    B_ptr,  # pointer to bias, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * stride
    X_ptr += row_idx * stride
    Mean_ptr += row_idx
    RSTD_ptr += row_idx

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    demeaned = X_row - mean
    var = tl.sum((demeaned) * (demeaned), axis=0) / n_cols
    rstd = rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    # Y_row = (demeaned) * rstd * W_row + B_row
    Y_row = tl.fma(demeaned * rstd, W_row, B_row)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DSc,  # pointer to the partial sum of weights gradient
                             DSh,  # pointer to the partial sum of biases gradient
                             Y,  # pointer to the input
                             Sc,  # pointer to the weights
                             Sh,  # pointer to the biases
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             GROUP_SIZE_M: tl.constexpr, 
                             BLOCK_SIZE_N: tl.constexpr
                             ):

    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    Y += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DSc = DSc + lock_id * N + cols
    DSh = DSh + lock_id * N + cols
    # Load data to SRAM
    y = tl.load(Y + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    sc = tl.load(Sc + cols, mask=mask).to(tl.float32)
    sh = tl.load(Sh + cols, mask=mask).to(tl.float32)
    # mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = (y - sh) / sc
    scdy = sc * dy
    xhat = tl.where(mask, xhat, 0.)
    scdy = tl.where(mask, scdy, 0.)
    c1 = tl.sum(xhat * scdy, axis=0) / N
    c2 = tl.sum(scdy, axis=0) / N
    dx = (scdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dsc = (dy * xhat).to(sc.dtype)
    partial_dsh = (dy).to(sc.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dsc += tl.load(DSc, mask=mask)
        partial_dsh += tl.load(DSh, mask=mask)
    tl.store(DSc, partial_dsc, mask=mask)
    tl.store(DSh, partial_dsh, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, 
                         BLOCK_SIZE_N: tl.constexpr
                         ):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)



class LNLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, shift, weight, bias, eps):
        ####
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        ####
        M, N = x.shape

        # do norm

        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x = x.view(-1, x.shape[-1])
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  #
            x, 
            y, 
            scale, 
            shift, 
            mean, 
            rstd,  #
            x.stride(0), 
            N, 
            eps,  #
            BLOCK_SIZE=BLOCK_SIZE, 
            num_warps=num_warps, 
            num_ctas=1
            )

        # now lets do linear
        ctx.save_for_backward(y, scale, shift, mean, rstd, weight, bias)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

        output = F.linear(y, weight, bias)

        ####
        output = output.view(orig_shape[:-1] + output.shape[-1:])
        ####
        
        return output


    @staticmethod
    def backward(ctx, do):
        y, scale, shift, m, v, weight, bias = ctx.saved_tensors

        ###
        orig_shape = do.shape
        do = do.view(-1, do.shape[-1])
        ###

        # Gradients w.r.t. Linear layer parameters
        # this is the one that progresses back to layer norm layer
        dy = F.linear(do, weight.transpose(-1,-2))
        dw = db = dscale = dshift = None
        if ctx.needs_input_grad[-3]:
            dw = F.linear(do.transpose(-1, -2), y.transpose(-1, -2))
        if ctx.needs_input_grad[-2] and bias is not None:
            db = do.sum(dim=0)

        # now do norm portion

        # heuristics for amount of parallel reduction stream for DW/DB
        N = orig_shape[-1]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=y.device)
        _dscale = torch.zeros((GROUP_SIZE_M, N), dtype=y.dtype, device=y.device)
        _dshift = torch.zeros((GROUP_SIZE_M, N), dtype=y.dtype, device=y.device)
        dscale = torch.empty((N, ), dtype=y.dtype, device=y.device)
        dshift = torch.empty((N, ), dtype=y.dtype, device=y.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        M, N = dy.shape

        _layer_norm_bwd_dx_fused[(M, )](  #
            dx, 
            dy, 
            _dscale, 
            _dshift, 
            y, 
            scale,
            shift,
            m, 
            v, 
            locks,  #
            y.stride(0), 
            N,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)

        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid](
            _dscale, _dshift, dscale, dshift, min(GROUP_SIZE_M, M), N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128, 
            num_ctas=1
            )

        ####
        dx = dx.reshape(orig_shape)
        ####

        return dx, dscale, dshift, dw, db, None


def test_ln_linear(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (2, M, N)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    norm = torch.nn.LayerNorm(N, eps=eps).to(device).to(dtype)
    linear = torch.nn.Linear(N, N).to(device).to(dtype)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)

    # forward pass
    y_tri = LNLinear.apply(x, norm.weight, norm.bias, linear.weight, linear.bias, eps)
    y_ref = linear(norm(x))

    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dsc_tri, dsh_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, norm.weight, norm.bias, linear.weight, linear.bias]]
    x.grad, norm.weight.grad, norm.bias.grad, linear.weight.grad, linear.bias.grad = None, None, None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dsc_ref, dsh_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, norm.weight, norm.bias, linear.weight, linear.bias]]
    # compare

    print("y_tri", y_tri)
    print("y_ref", y_ref)
    print("dx_tri", dx_tri)
    print("dx_ref", dx_ref)
    print("dsc_tri", dsc_tri)
    print("dsc_ref", dsc_ref)
    print("dsh_tri", dsh_tri)
    print("dsh_ref", dsh_ref)
    print("dw_tri", dw_tri)
    print("dw_ref", dw_ref)
    print("db_tri", db_tri)
    print("db_ref", db_ref)

    diff_y = torch.max(torch.abs(y_tri - y_ref))
    diff_dx_max = torch.max(torch.abs(dx_tri - dx_ref))
    diff_dsc_max = torch.max(torch.abs(dsc_tri - dsc_ref))
    diff_dsh_max = torch.max(torch.abs(dsh_tri - dsh_ref))
    diff_db_max = torch.max(torch.abs(db_tri - db_ref))
    diff_dw_max = torch.max(torch.abs(dw_tri - dw_ref))

    print("diff_y", diff_y, "y_norm", torch.norm(y_ref))
    print("diff_dx_max", diff_dx_max, "dx_norm", torch.norm(dx_ref))
    print("diff_dsc_max", diff_dsc_max, "dsc_norm", torch.norm(dsc_ref))
    print("diff_dsh_max", diff_dsh_max, "dsh_norm", torch.norm(dsh_ref))
    print("diff_db_max", diff_db_max, "db_norm", torch.norm(db_ref))
    print("diff_dw_max", diff_dw_max, "dw_norm", torch.norm(dw_ref))

    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dsc_tri, dsc_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dsh_tri, dsh_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 32, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_ln_linear(M, N, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (2, M, N)
    norm = torch.nn.LayerNorm(N, eps=eps).to(device).to(dtype)
    linear = torch.nn.Linear(N, N).to(device).to(dtype)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return LNLinear.apply(x, norm.weight, norm.bias, linear.weight, linear.bias, eps)

        if provider == "torch":
            return linear(norm(x))

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    # test_ln_linear(1152, 8192, torch.float16)
    bench_ln_linear.run(save_path='.', print_data=True)
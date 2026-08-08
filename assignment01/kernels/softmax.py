"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_x_row, stride_x_col,
    stride_y_row, stride_y_col,
    BLOCK_N: tl.constexpr  # 必须标记为 tl.constexpr
):
    # 1. 获取当前 Program 处理的行号 (单个 Program 处理 1 行)
    row_idx = tl.program_id(0)
    
    # 2. 计算当前行的内存基地址和列偏移
    offs_n = tl.arange(0, BLOCK_N)
    x_row_ptr = x_ptr + row_idx * stride_x_row
    y_row_ptr = y_ptr + row_idx * stride_y_row
    
    # 3. 构造 1D 列掩码（防止 N 越界）
    mask = offs_n < N
    
    # 4. 加载整行数据，越界位置补 -inf（处理数值极大/越界问题）
    input_ptrs = x_row_ptr + offs_n * stride_x_col
    x = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    
    # 5. 行内数值稳定 Softmax 计算
    row_max = tl.max(x, axis=0)             # 求整行最大值
    exp_stable = tl.exp(x - row_max)         # 减最大值后再 exp
    sum_val = tl.sum(exp_stable, axis=0)     # 整行求和
    output = exp_stable / sum_val
    
    # 6. 使用 y 的 stride 存回内存
    output_ptrs = y_row_ptr + offs_n * stride_y_col
    tl.store(output_ptrs, output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    
    # 计算大于等于 N 的最小 2 的幂次作为 Block 大小
    BLOCK_N = triton.next_power_of_2(N)
    
    # 分配输出内存
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # 一个 program 处理一行，Grid 维度直接为 (M, )
    grid = (M, )
    
    # 调用 Kernel
    softmax_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_N=BLOCK_N
    )
    return y
"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T

def make_softmax(M, N, BLOCK_M, BLOCK_N, 
                 threads=128, num_stages=3,
                 dtype="float32"):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Y: T.Tensor((M,N), dtype)):
        with T.Kernel(
          T.ceildiv(N, BLOCK_N), 
          T.ceildiv(M, BLOCK_M), 
          threads = threads
        ) as (bx, by):
          x_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
          max_local = T.alloc_fragment((BLOCK_M,), dtype)
          sum_local = T.alloc_fragment((BLOCK_M,), dtype)
          T.clear(max_local)
          T.clear(sum_local)

          # copy X to x_local and padding -inf for ineffective items
          for i,j in T.Parallel(BLOCK_M, BLOCK_N):
            gi = by * BLOCK_M + i
            gj = j
            valid = gi < M and gj < N
            x_local[i, j] = T.if_then_else(
              valid,
              X[gi, gj],
              -T.infinity(dtype)
            )

          # compute row maximum
          T.reduce_max(x_local, max_local, dim=1)
          
          # exp(x-max) and zero padding
          for i,j in T.Parallel(BLOCK_M, BLOCK_N):
            gi = by * BLOCK_M + i
            gj = j
            valid = gi < M and gj < N
            x_local[i,j] = T.if_then_else(
               valid,
               T.exp(x_local[i,j] - max_local[i]),
               0
            )

          # local sum
          T.reduce_sum(x_local, sum_local, dim=1)

          # normalize and storei
          for i,j in T.Parallel(BLOCK_M, BLOCK_N):
            gi = by * BLOCK_M + i
            gj = j
            valid = gi < M and gj < N
            if valid:
              Y[gi, gj] = x_local[i,j] / sum_local[i] 
    return main      
    

def softmax(x: torch.Tensor) -> torch.Tensor:
    # raise NotImplementedError("从这里开始写")
    M, N = x.shape
    BLOCK_M = 4
    BLOCK_N = 2
    while (BLOCK_N < N):
      BLOCK_N *= 2
    func = make_softmax(M, N, BLOCK_M, BLOCK_N)
    kernel = tilelang.compile(func, out_idx=[1])
    y = kernel(x)
    return y



